import chainlit as cl
import torch
import psycopg2
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F
import os

os.environ["CHAINLIT_DATABASE"] = "local"

# Конфигурация
DB_CONN = "dbname=appdb user=appuser password=secret port=5432 host=rag-data"
EMB_MODEL_PATH = "/wrk/models/embedding_models/models--intfloat--multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439"
LLM_MODEL_PATH = "/wrk/models/llms/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
TOP_K = 5
MAX_LENGTH = 512

# Инициализация моделей (будет выполнена один раз при старте)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_tokenizer = None
emb_model = None
llm_tokenizer = None
llm_model = None
conn = None

def average_pool(last_hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings/sum_mask

@torch.inference_mode()
def embed(text: str):
    inputs = emb_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)
    
    outputs = emb_model(**inputs)
    emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()

def search_context(query, top_k=TOP_K):
    try:
        query_emb = embed(query).tolist()
        cur = conn.cursor()
        cur.execute(
            "SELECT content, metadata FROM documents_test ORDER BY embedding <-> %s LIMIT %s",
            (json.dumps(query_emb), top_k)
        )
        results = cur.fetchall()
        cur.close()
        return [r[0] for r in results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def ask_llm(question, context, chat_history):
    prompt = f'''
Ты - умный ассистент, помогающий сотрудникам ответить на вопросы по документам. Используй приведённый контекст для ответа на вопросы.
Если ответ не найден в контексте - скажи, что информации нет.

История диалога:
{chat_history}

Контекст из документов:
{context}

Вопрос пользователя:
{question}

Отвечай строго на вопрос пользователя.
'''
    
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
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
    answer = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer

def rephrase_question(question, history):
    if not history:
        return question
    
    history_text = "\n".join([f"Пользователь: {h['user']}\nАссистент: {h['assistant']}" for h in history])
    prompt = f"""
История диалога:
{history_text}

Вопрос пользователя: "{question}"

Переформулируй вопрос, учитывая предыдущий контекст, чтобы он был самодостаточным запросом для поиска в документах.
Не уходи от темы и используй только данный контекст в виде истории диалога.
"""
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
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

@cl.on_chat_start
async def on_chat_start():
    # Инициализация моделей при старте чата
    global emb_tokenizer, emb_model, llm_tokenizer, llm_model, conn
    
    msg = cl.Message(content="Инициализация моделей...")
    await msg.send()
    
    try:
        # Загрузка моделей
        emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_PATH)
        emb_model = AutoModel.from_pretrained(EMB_MODEL_PATH).to(device)
        emb_model.eval()
        
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        llm_model.eval()
        
        # Подключение к БД
        conn = psycopg2.connect(DB_CONN)
        
        # Сохраняем историю в пользовательской сессии
        cl.user_session.set("chat_history", [])
        
        msg.content = "Модели загружены! Задайте ваш вопрос."
        await msg.update()
        
    except Exception as e:
        msg.content = f"Ошибка инициализации: {str(e)}"
        await msg.update()
        raise e

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history", [])
    
    msg = cl.Message(content="Обрабатываю ваш запрос...")
    await msg.send()
    
    try:
        # Рефразирование вопроса с учетом истории
        if chat_history:
            new_question = rephrase_question(message.content, chat_history)
        else:
            new_question = message.content
        
        # Поиск релевантного контекста
        context_chunks = search_context(new_question)
        context = "\n\n".join(context_chunks)
        
        # Генерация ответа
        answer = ask_llm(message.content, context, chat_history)
        
        # Обновление истории
        chat_history.append({'user': message.content, 'assistant': answer})
        if len(chat_history) > 10:  # Ограничиваем историю
            chat_history = chat_history[-10:]
        cl.user_session.set("chat_history", chat_history)
        
        # Отправка ответа
        msg.content = answer
        await msg.update()
        
        # Дополнительно: показываем источники
        # if context_chunks:
        #     sources_msg = cl.Message(content=f"Найдено {len(context_chunks)} релевантных документов")
        #     await sources_msg.send()
    
    except Exception as e:
        msg.content = f"Произошла ошибка: {str(e)}"
        await msg.update()

@cl.on_chat_end
def on_chat_end():
    # Очистка ресурсов при завершении чата
    if conn:
        conn.close()
    torch.cuda.empty_cache()

# Запуск приложения
if __name__ == "__main__":
    # Запуск: chainlit run chainlit.py
    pass