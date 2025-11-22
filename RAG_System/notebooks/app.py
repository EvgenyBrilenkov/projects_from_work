import os
import json
import psycopg2
import torch
import torch.nn.functional as F
import chainlit as cl
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from chainlit.element import File
from datetime import datetime
from typing import Optional
from chainlit.types import ThreadDict
from uuid import uuid4
from starlette.staticfiles import StaticFiles
import asyncio
from chainlit.server import app
from starlette.routing import Mount
import time
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

# ------------------------
# Config
# ------------------------
DB_CONN = "dbname=appdb user=appuser password=secret port=5433 host=10.101.10.106"
CHAINLIT_CONN = "postgresql+asyncpg://appuser:secret@10.101.10.106:5433/appdb"
EMB_MODEL_PATH = "/wrk/models/models--intfloat--multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439"
LLM_MODEL_PATH = "/wrk/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
DOCS = "/wrk/data"
TOP_K = 9
inference_semaphore = asyncio.Semaphore(1)

# ------------------------
# Models
# ------------------------
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_PATH)
emb_model = AutoModel.from_pretrained(EMB_MODEL_PATH).to(device).eval()

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    device_map=device,
    trust_remote_code=True,
    quantization_config=quantization_config,
    #dtype = torch.float16
).eval()
llm_model = torch.compile(llm_model)

# ------------------------
# DB
# ------------------------
def get_db_connection():
    return psycopg2.connect(DB_CONN)

app.router.routes.insert(0, Mount("/docs", app=StaticFiles(directory=DOCS), name="docs"))

# ------------------------
# Embedding helpers
# ------------------------
MAX_LENGTH = 512

def average_pool(last_hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def embed(text: str):
    inputs = emb_tokenizer("query: "+text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
        emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()

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
            SELECT doc_id, content, metadata, 1 - (embedding <-> %s) AS vec_score
            FROM documents
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (json.dumps(query_emb), json.dumps(query_emb), top_k*2)
        )
        vec_results = cur.fetchall()

        # BM25
        cur.execute(
            """
            SELECT doc_id, content, metadata, ts_rank_cd(tsv, plainto_tsquery('russian', %s)) AS bm25_score
            FROM documents
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

        for rank, (doc_id, content, metadata, _) in enumerate(vec_results):
            key = content
            rrf_scores.setdefault(key, {'doc_id': doc_id, 'metadata': metadata, 'score':0})
            rrf_scores[key]['score'] += 1.0 / (k + rank + 1)

        for rank, (doc_id, content, metadata, _) in enumerate(bm25_results):
            key = content
            if key not in rrf_scores:
                rrf_scores[key] = {'doc_id': doc_id, 'metadata': metadata, 'score':0}
            rrf_scores[key]['score'] += 1.0 / (k + rank + 1)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        return [(item['doc_id'], content, item['metadata']) for content, item in sorted_items][:top_k]
    
    finally:
        cur.close()
        conn.close()

# ------------------------
# LLM helpers
# ------------------------
def ask_llm(question, context, chat_history):
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
    
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.inference_mode():
        output = llm_model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=terminators,
            num_beams=1
        )
        
    generated_tokens = output[0][input_ids["input_ids"].shape[-1]:]    
    answer = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer

def rephrase_question(question, history):
    history_text = "\n".join([f"Пользователь: {h['user']}\nАссистент: {h['assistant']}" for h in history])

    system_prompt = """Ты - помощник для переформулирования поисковых запросов. 
Переформулируй последний вопрос пользователя с учетом контекста диалога, 
но НЕ включай информацию из предыдущих ответов, если ты не смог найти ответ на вопрос, в новый поисковый запрос."""

    prompt = f"""История диалога (только для контекста):
{history_text}

Текущий вопрос: 
{question}

Переформулируй текущий вопрос как самодостаточный поисковый запрос, 
сохраняя его оригинальный смысл. Не упоминай предыдущие нерелевантные ответы.

Переформулированный вопрос:
"""
    messages = [{"role":"system", "content":system_prompt},{"role":"user", "content":prompt}]

    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.inference_mode():
        output = llm_model.generate(
            **input_ids,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=terminators,
            num_beams=1
        )
    
    generated_tokens = output[0][input_ids["input_ids"].shape[-1]:]    
    return llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# ------------------------
# Chat history test
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
    cl.user_session.set("chat_history", [])

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

    msg = await cl.Message(content="⌛ Ваш запрос в очереди на исполнение. Пожалуйста, подождите...", author="Assistant").send()
    chat_history = cl.user_session.get("chat_history", [])
    loop = asyncio.get_event_loop()

    try:
        # Рефразирование вопроса с учетом истории
        old_q = message.content
        async with inference_semaphore:
            if chat_history:
                rephrase_task = loop.run_in_executor(None, rephrase_question, message.content, chat_history)
                new_question = await run_with_dots(msg, "♻️ Формирую запрос", rephrase_task)
                rephrased_q = new_question
            else:
                msg.content = "⌛ Ваш запрос в очереди на исполнение. Пожалуйста, подождите..."
                await msg.update()
                new_question = message.content
                rephrased_q = None

        # Поиск релевантного контекста
        async with inference_semaphore:
            search_task = loop.run_in_executor(None, search_context, new_question)
            context_chunks = await run_with_dots(msg, "🔍 Ищу релевантные документы", search_task)
        if not context_chunks:
            msg.content = "❌ Информация в документах не найдена."
            await msg.update()

        # Достаём текст чанков
        context = "\n\n".join([c[1] for c in context_chunks])
        doc_id = context_chunks[0][0]
        sources = [c[2]['путь'] for c in context_chunks]
        
        # Генерация ответа
        async with inference_semaphore:
            llm_task = loop.run_in_executor(None, ask_llm, message.content, context, chat_history)
            answer = await run_with_dots(msg, "✍️ Генерирую ответ", llm_task)
        
        data_layer = get_data_layer()
        
        paths = []
        files = []
        for fp in sources:
            if fp not in paths:
                try:
                    display_name = os.path.basename(fp)
                    ending = display_name[-4::]
                    if "pdf" in ending or "pptx" in ending:
                        files.append(f"🔴📃 [{display_name}](/docs/{display_name})")
                        paths.append(fp)
                    elif "doc" in ending or "txt" in ending:
                        files.append(f"🔵📃 [{display_name}](/docs/{display_name})")
                        paths.append(fp)
                    else:
                        files.append(f"🟢📃 [{display_name}](/docs/{display_name})")
                        paths.append(fp)

                except Exception as e:
                    print(f"❌ Не удалось прикрепить файл {fp}: {e}")

        # Отправка ответа
        sources_text = "\n".join(f"- {link}" for link in files)
        end_time = time.time()
        msg.content=f"{answer}\n\n⌛ Время исполнения запроса: {round(end_time-start_time, 1)} секунд.\n\n📁 Источники:\n{sources_text}"
        await msg.update()

        # Обновление истории
        chat_history.append({'user': message.content, 'assistant': answer, 'doc_id': doc_id})
        if len(chat_history) > 10:  # Ограничиваем историю
            chat_history = chat_history[-10:]
        cl.user_session.set("chat_history", chat_history)
        timestamp = datetime.now()
        
        user_id = "1"

        context = "\n-----------------------------------------------------------\n".join([c[1] for c in context_chunks])
        save_chat_history(user_id, doc_id, old_q, rephrased_q, answer, timestamp, sources, context)

    except Exception as e:
        msg = cl.Message(content="♻️ Обрабатываю запрос...", author="Assistant")
        msg.content=f"☠️ Произошла ошибка: {str(e)}"
        await msg.send()
        
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass

@cl.on_chat_end
def on_chat_end():
    torch.cuda.empty_cache()

# Запуск приложения
if __name__ == "__main__":
    pass