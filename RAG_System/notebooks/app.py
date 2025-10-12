# Настройте конфиг (укажите пути к моделям и подключение к БД)
# Запустите скрипт через консоль chainlit run app.py -w

import os
import json
import psycopg2
import torch
import torch.nn.functional as F
import chainlit as cl
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict
from scipy.spatial import distance
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from chainlit.data import BaseDataLayer
from chainlit.element import ElementDict, File
from chainlit.step import StepDict
from chainlit import User, PersistedUser
from chainlit.types import ThreadDict, Pagination, ThreadFilter, PaginatedResponse, PageInfo
from typing import Optional, Dict, List

# ------------------------
# Config
# ------------------------
DB_CONN = "dbname=appdb user=appuser password=secret port=5432 host=rag-data" # Подключение к вашей БД
EMB_MODEL_PATH = "/wrk/models/embedding_models/models--intfloat--multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439" # Путь до вашей эмбеддинговой модели
LLM_MODEL_PATH = "/wrk/models/llms/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a" # Путь до вашей языковой модели
TOP_K = 7 # Сколько релевантных чанков для вашего запроса будет извлекать эмбеддинговая модель

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
    # torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
).eval()

# ------------------------
# DB
# ------------------------
conn = psycopg2.connect(DB_CONN)
cur = conn.cursor()

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
    inputs = emb_tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
        emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()

# ------------------------
# DB search
# ------------------------
def search_context(query, top_k=TOP_K):
    query_emb = embed(query).tolist()
    cur.execute(
        """
        SELECT doc_id, content, metadata FROM documents_e5
        ORDER BY embedding <-> %s
        LIMIT %s
        """,
        (json.dumps(query_emb), top_k)
    )
    results = cur.fetchall()
    return results  # [(doc_id, chunk, metadata), ...]

# ------------------------
# LLM helpers
# ------------------------
def ask_llm(question, context, chat_history):
    prompt = f'''
Ты - ассистент для поиска в документах. Отвечай ТОЛЬКО на основе предоставленного контекста.

КОНТЕКСТ ДОКУМЕНТОВ:
{context}

ИСТОРИЯ ДИАЛОГА:
{chat_history}

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе контекста выше
2. Если информации нет в контексте - скажи "В предоставленных документах нет информации по этому вопросу"
3. Будь точным и лаконичным
4. Не придумывай информацию
5. Если нужно, уточни какой документ используешь

ВОПРОС: {question}

ОТВЕТ:'''

    messages = [{"role":"user", "content":prompt}]
    
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generaton_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True
    ).to(device)
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        output = llm_model.generate(
            **input_ids,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=terminators,
            num_beams=1
        )
        
    generated_tokens = output[0][input_ids["input_ids"].shape[-1]:]    
    answer = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer

def rephrase_question(question, history):
    history_text = "\n".join([f"Пользователь: {h['user']}\nАссистент: {h['assistant']}" for h in history])
    prompt = f"""
Ты помощник для переформулирования поисковых запросов. 
Переформулируй последний вопрос пользователя с учетом контекста диалога, 
но НЕ включай информацию из предыдущих ответов, если ты не смог найти ответ на вопрос, в новый поисковый запрос.

История диалога (только для контекста):
{history_text}

Текущий вопрос: "{question}"

Переформулируй текущий вопрос как самодостаточный поисковый запрос, 
сохраняя его оригинальный смысл. Не упоминай предыдущие нерелевантные ответы.

Переформулированный вопрос:
"""
    messages = [{"role":"user", "content":prompt}]

    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generaton_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True
    ).to(device)
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        output = llm_model.generate(
            **input_ids,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=terminators,
            num_beams=1
        )
    
    generated_tokens = output[0][input_ids["input_ids"].shape[-1]:]    
    return llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# ------------------------
# Chat history test
# ------------------------
chat_history = []

def save_chat_history(user_id, doc_id, user_msg, rephrased_msg, assistant_msg, sources_ids):
    cur.execute(
        """
        INSERT INTO chat_history (user_id, doc_id, user_message, rephrased_message, assistant_message, sources_ids)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (user_id, doc_id, user_msg, rephrased_msg, assistant_msg, sources_ids)
    )
    conn.commit()

# ------------------------
# Chainlit core
# ------------------------
@cl.on_chat_start
async def on_chat_start():
    try:
        # Сохраняем историю в пользовательской сессии
        cl.user_session.set("chat_history", [])
        
    except Exception as e:
        msg = cl.Message(content=f"Ошибка инициализации: {str(e)}")
        await msg.update()
        raise e
    
@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history", [])
    
    msg = cl.Message(content="Обрабатываю ваш запрос...")
    await msg.send()

    try:
        # Рефразирование вопроса с учетом истории
        old_q = message.content
        if chat_history:
            new_question = rephrase_question(message.content, chat_history)
            rephrased_q = new_question
        else:
            new_question = message.content
            rephrased_q = None

        # Поиск релевантного контекста
        context_chunks = search_context(new_question)
        if not context_chunks:
            msg.content = "Информация в документах не найдена."
            await msg.update()
            return

        # Достаём текст чанков
        context = "\n\n".join([c[1] for c in context_chunks])
        doc_id = context_chunks[0][0]  # Сохраняем doc_id из первого совпадения
        sources = [f"{c[0]} ({c[2]['путь']})" for c in context_chunks]
        sources = list(dict.fromkeys(sources))
        
        # Генерация ответа
        answer = ask_llm(message.content, context, chat_history)

        # Обновление истории
        chat_history.append({'user': message.content, 'assistant': answer, 'doc_id': doc_id})
        if len(chat_history) > 10:  # Ограничиваем историю
            chat_history = chat_history[-10:]
        cl.user_session.set("chat_history", chat_history)
        
        user_id = "1"

        save_chat_history(user_id, doc_id, old_q, rephrased_q, answer, sources)
        
        # Отправка ответа
        msg.content = f"{answer}\n\nИсточники:\n\n"
        await msg.update()
        
        
        for src in sources:
            file_path = src.split("(")[-1].rstrip(")")
            file_path = file_path.strip(" )")
            if os.path.exists(file_path):
                await File(name=os.path.basename(file_path), path=file_path).send(for_id=msg.id)

    except Exception as e:
        msg.content = f"Произошла ошибка: {str(e)}"
        await msg.update()
@cl.on_chat_resume
async def on_chat_resume(thread):
    pass

@cl.on_chat_end
def on_chat_end():
    # Очистка ресурсов при завершении чата
    torch.cuda.empty_cache()

# Запуск приложения
if __name__ == "__main__":
    # Запуск: chainlit run app.py -w
    pass
