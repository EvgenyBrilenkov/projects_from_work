import os
import json
import psycopg2
import torch
import torch.nn.functional as F
import chainlit as cl
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from typing import Optional
from chainlit.types import ThreadDict
from starlette.staticfiles import StaticFiles
import asyncio
from chainlit.server import app
from starlette.routing import Mount
import time
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from urllib.parse import quote
import traceback
from torch import Tensor
from ollama import AsyncClient

#warnings.filterwarnings("ignore")

# ------------------------
# Config
# ------------------------
DB_CONN = "dbname=appdb user=appuser password=secret port=5432 host=db"
CHAINLIT_CONN = "postgresql+asyncpg://appuser:secret@db:5432/appdb"
EMB_MODEL_PATH = "/wrk/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
DOCS = "/wrk/data/demo_data_sql"
TOP_K = 9
inference_semaphore = asyncio.Semaphore(1)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
# ------------------------
# Models
# ------------------------
torch.cuda.empty_cache() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_PATH)
emb_model = AutoModel.from_pretrained(EMB_MODEL_PATH).to(device).eval()

# ------------------------ 
# DB
# ------------------------
def get_db_connection():
    return psycopg2.connect(DB_CONN)

app.router.routes.insert(0, Mount("/docs", app=StaticFiles(directory=DOCS), name="docs"))

# ------------------------
# Embedding helpers
# ------------------------
MAX_LENGTH = 1024

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def embed(text: str):
    query = f"Instruct: Учитывая поисковый запрос, найди релевантные фрагменты текста, которые отвечают на него\nQuery: {text}"

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = emb_tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)
        outputs = emb_model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy()

# ------------------------
# DB search
# ------------------------
def search_context(query, top_k=TOP_K):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Embeddings
        query_emb = embed(query).tolist()
        cur.execute(
            """
            SELECT doc_id, content, 1 - (embedding <-> %s) AS vec_score
            FROM demo_data
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (json.dumps(query_emb), json.dumps(query_emb), top_k*2)
        )
        vec_results = cur.fetchall()

        # BM25
        cur.execute(
            """
            SELECT doc_id, content, ts_rank_cd(tsv, plainto_tsquery('russian', %s)) AS bm25_score
            FROM demo_data
            WHERE tsv @@ plainto_tsquery('russian', %s)
            ORDER BY bm25_score DESC
            LIMIT %s
            """,
            (query, query, top_k*2)
        )
        bm25_results = cur.fetchall()

        # Ranking and concat
        k = 60
        rrf_scores = {}

        for rank, (doc_id, content, _) in enumerate(vec_results):
            key = (doc_id, content)
            rrf_scores.setdefault(key, {'score':0})
            rrf_scores[key]['score'] += 1.0 / (k + rank + 1)

        for rank, (doc_id, content, _) in enumerate(bm25_results):
            key = (doc_id, content)
            if key not in rrf_scores:
                rrf_scores[key] = {'score':0}
            rrf_scores[key]['score'] += 1.0 / (k + rank + 1)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        top_chunks = [(doc_id, content) for (doc_id, content), _ in sorted_items[:top_k]]

        # Searching sources
        unique_doc_ids = list(set(doc_id for doc_id, _ in top_chunks))
        placeholders = ','.join(['%s'] * len(unique_doc_ids))

        cur.execute(
            f"""
            SELECT doc_id, metadata 
            FROM demo_data_metadata 
            WHERE doc_id IN ({placeholders})
            """,
            unique_doc_ids
        )

        metadata_map = {row[0]: row[1] for row in cur.fetchall()}

        result = []
        for doc_id, content in top_chunks:
            metadata = metadata_map.get(doc_id, {})
            result.append((doc_id, content, metadata))

        return result
    
    finally:
        cur.close()
        conn.close()

# ------------------------
# LLM helpers
# ------------------------
async def ask_llm(question, context, chat_history):
    system_prompt = """Ты - ассистент для поиска в документах. Отвечай ТОЛЬКО на основе предоставленного контекста.

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе контекста ниже
2. Если информации нет в контексте - скажи "В предоставленных документах нет информации по этому вопросу"
3. Будь точным
4. Не придумывай информацию
5. Если нужно, уточни какой документ используешь    
"""

    prompt = f"""КОНТЕКСТ ДОКУМЕНТОВ:
{context}

ИСТОРИЯ ДИАЛОГА:
{chat_history}

ВОПРОС: 
{question}

ОТВЕТ:"""

    messages = [{"role":"system", "content":system_prompt},{"role":"user", "content":prompt}]
    
    client = AsyncClient(host=OLLAMA_BASE_URL)
    
    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0.05,
            "num_predict": 1024,
            "top_p":0.95
        },
        keep_alive="3m"
    )

    return response["message"]["content"].strip()

async def rephrase_question(question, history):
    history_text = "\n".join([f"Пользователь: {h['user']}\nАссистент: {h['assistant']}" for h in history])

    system_prompt = """Ты - помощник для переформулирования поисковых запросов. 
Переформулируй последний вопрос пользователя с учетом контекста диалога, но НЕ включай информацию из предыдущих ответов, если ты не смог найти ответ на вопрос, в новый поисковый запрос."""

    prompt = f"""История диалога (только для контекста):
{history_text}

Текущий вопрос: 
{question}

Переформулируй текущий вопрос как самодостаточный поисковый запрос, сохраняя его оригинальный смысл, но с упором на текущий вопрос. Не упоминай предыдущие нерелевантные ответы.

Переформулированный вопрос:
"""
    messages = [{"role":"system", "content":system_prompt},{"role":"user", "content":prompt}]

    client = AsyncClient(host=OLLAMA_BASE_URL)
    
    response = await client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
            "temperature": 0.05,
            "num_predict": 300,
            "top_p":0.95
        },
            keep_alive="3m"
        )

    return response["message"]["content"].strip()

# ------------------------
# Chat history
# ------------------------
def save_chat_history(user_id, doc_id, user_msg, rephrased_msg, assistant_msg, timestamp, sources_ids, chunks):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO chat_history (user_id, doc_id, user_message, rephrased_message, assistant_message, timestamp, sources_ids, chunks)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, doc_id, user_msg, rephrased_msg, assistant_msg, timestamp, sources_ids, chunks)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def load_chat_history(thread_id: str, max_pairs: int = 20):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT type, output FROM steps WHERE "threadId" = %s AND type IN ('user_message', 'assistant_message') AND output is NOT NULL ORDER BY "createdAt" ASC
            """, (thread_id,))

        rows = cur.fetchall()
        chat_history = []
        user_msg = None

        ignore_prefixes = ("⌛", "♻️", "🔍", "✍️", "❌")

        for msg_type, content in rows:
            if msg_type == 'user_message':
                user_msg = content
            elif msg_type == 'assistant_message' and user_msg and not any(content.startswith(p) for p in ignore_prefixes):
                chat_history.append({"user": user_msg, "assistant": content.split("⌛")[0][:-2:]})
                user_msg = None
        
        return chat_history[-max_pairs:]
    except:
        return ''

    finally:
        cur.close()
        conn.close()

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=CHAINLIT_CONN
    )

def get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

@cl.on_chat_start
async def start_chat():
    torch.cuda.empty_cache()

@cl.password_auth_callback
async def on_login(username: str, password: str) -> Optional[cl.User]:
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor()
        cur.execute("SELECT id, identifier, metadata FROM users WHERE metadata->>'username' = %s;", (username,))
        row = cur.fetchone()
        if row:
            user_id, identifier, metadata = row
            if metadata.get("password") == password:
                return cl.User(identifier=identifier, display_name=metadata.get("display_name"), metadata={"username":metadata.get("username"), "password":metadata.get("password"), "access":metadata.get("access"), "display_name":metadata.get("display_name")})
    finally:
        if conn:
            cur.close()
            conn.close()
    return None

# ------------------------
# Chainlit core
# ------------------------
async def run_with_dots(
    message: cl.Message,
    base_text: str,
    task: asyncio.Task,
    dots_interval: float=0.6,
    max_dots: int=3):
    dots = ""
    while not task.done():
        dots = "." * (len(dots)%max_dots+1)
        message.content = f"{base_text}{dots}"
        await message.update()
        await asyncio.sleep(dots_interval)
    return await task

@cl.on_message
async def on_message(message: cl.Message):
    start_time = time.time()
    loop = asyncio.get_event_loop()
    thread_id = cl.context.session.thread_id
    chat_history = load_chat_history(thread_id, max_pairs=20)

    queue_position = inference_semaphore._value
    total_slots = 1
    waiting_count = total_slots - queue_position
    
    if waiting_count > 0:
        initial_msg = f"⌛ Ваш запрос в очереди. Ожидайте..."
    else:
        initial_msg = "♻️ Начинаю обработку..."
    
    msg = await cl.Message(content=initial_msg, author="Assistant").send()

    try:
        async with inference_semaphore:
            msg.content = "♻️ Начинаю обработку..."
            await msg.update()

            old_q = message.content
            if chat_history:
                rephrase_task = asyncio.create_task(rephrase_question(message.content, chat_history))
                new_question = await run_with_dots(msg, "♻️ Формирую запрос", rephrase_task)
                rephrased_q = new_question
            else:
                new_question = message.content
                rephrased_q = None

            search_task = loop.run_in_executor(None, search_context, new_question)
            context_chunks = await run_with_dots(msg, "🔍 Ищу релевантные документы", search_task)

            if not context_chunks:
                msg.content = "❌ Информация в документах не найдена."
                await msg.update()
                return

            context = "\n\n".join([c[1] for c in context_chunks])
            doc_id = context_chunks[0][0]
            
            llm_task = asyncio.create_task(ask_llm(message.content, context, chat_history))
            answer = await run_with_dots(msg, "✍️ Генерирую ответ", llm_task)
            answer = answer.replace("\\n", "\n")

            sources = [c[2]['path'] for c in context_chunks]
            paths = []
            files = []
            for fp in sources:
                if fp not in paths:
                    try:
                        display_name = os.path.basename(fp)
                        ending = display_name[-4:].lower()
                        link_ = quote(display_name)
                        if "pdf" in ending or "pptx" in ending:
                            files.append(f"🔴📃 [{display_name}](/docs/{link_})")
                            paths.append(fp)
                        elif "doc" in ending or "docx" in ending or "txt" in ending or "md" in ending:
                            files.append(f"🔵📃 [{display_name}](/docs/{link_})")
                            paths.append(fp)
                        else:
                            files.append(f"🟢📃 [{display_name}](/docs/{link_})")
                            paths.append(fp)
                    except Exception as e:
                        print(f"❌ Не удалось прикрепить файл {fp}: {e}")

            sources_text = "\n".join(f"- {link}" for link in files[:5])
            end_time = time.time()
            msg.content = f"{answer}\n\n⌛ Время исполнения запроса: {round(end_time - start_time, 1)} секунд.\n\n📁 Источники:\n{sources_text}"
            await msg.update()

            timestamp = datetime.now()
            current_user = cl.user_session.get("user")
            user_id = current_user.identifier if current_user else "anonymous"
            context_for_save = "\n-----------------------------------------------------------\n".join([c[1] for c in context_chunks])
            save_chat_history(user_id, doc_id, old_q, rephrased_q, answer, timestamp, sources, context_for_save)

    except Exception as e:
        print(traceback.format_exc())
        await cl.Message(content=f"☠️ Произошла ошибка: {str(e)}", author="Assistant").send()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass

@cl.on_chat_end
def on_chat_end():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    pass