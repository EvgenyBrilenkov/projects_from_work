# Как создать свою БД:  
# - Скачайте PostgreSQL и pgvector  
# - С помощью команд ниже подключите расширение vector и создайте таблицу demo_data

# PostgreSQL: (в cmd)  
# psql -h localhost -d appdb -U appuser  
# CREATE EXTENSION IF NOT EXISTS vector;  
# CREATE TABLE demo_data (id SERIAL PRIMARY KEY, doc_id TEXT, chunk_id INT, content TEXT, embedding vector(1024));  
# CREATE TABLE demo_data (id SERIAL PRIMARY KEY, doc_id TEXT, metadata JSONB);

# Также создайте таблицу chat_history для будущего логирования запросов.  
# CREATE TABLE chat_history (id SERIAL PRIMARY KEY, user_id TEXT, doc_id TEXT, user_message TEXT, rephrased_message TEXT, assistant_message TEXT, timestamp TIMESTAMP, sources_ids TEXT, chunks TEXT); 

from transformers import AutoModel, AutoTokenizer
import torch
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions, PictureDescriptionVlmOptions 
import re
from chonkie import Pipeline
import tempfile
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from datetime import datetime
import json
import psycopg2
from pathlib import Path
from tqdm import tqdm

# Config
FOLDER_PATH = "/wrk/data/norma_cs_data"
DB_CONN = "dbname=appdb user=appuser password=secret port=5432 host=db"
MODEL_PATH = "/wrk/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"

emb_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
emb_model = AutoModel.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_model.eval()

def get_converters(use_ocr=False, picture_desc=False):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_picture_description = picture_desc

    if use_ocr:
        pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=["rus", "eng"])
    if picture_desc:
        pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
            repo_id="Qwen/Qwen3-VL-2B-Instruct",
            prompt="Напиши описание изображения на русском языке.",
        )
        pipeline_options.images_scale = 1
        pipeline_options.generate_picture_images = False

    word_converter = DocumentConverter(format_options={InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options)})
    main_converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    return word_converter, main_converter

TABLE_LABEL_RE = re.compile(
    r"^(Таблица|Table)\s+.+$",
    re.IGNORECASE,
)


def get_chunk_text(chunk) -> str:
    return chunk.text if hasattr(chunk, "text") else str(chunk)


def set_chunk_text(chunk, text: str):
    if hasattr(chunk, "text"):
        chunk.text = text
        return chunk

    return text


def is_table_chunk(text: str) -> bool:

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        return False

    pipe_lines = sum("|" in line for line in lines)

    has_separator = any(
        re.match(r"^\|?[\s:\-|]+\|?$", line)
        for line in lines
    )

    return pipe_lines >= 2 and has_separator


def extract_table_label_from_start(text: str) -> str | None:

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return None

    line = lines[0]

    if TABLE_LABEL_RE.match(line):
        return line

    return None


def extract_table_label_from_end(text: str) -> str | None:

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return None

    line = lines[-1]

    if TABLE_LABEL_RE.match(line):
        return line

    return None


def patch_table_chunks(chunks):

    patched = []

    for i, chunk in enumerate(chunks):

        text = get_chunk_text(chunk)

        if not is_table_chunk(text):
            patched.append(chunk)
            continue

        if not patched:
            patched.append(chunk)
            continue

        previous_text = get_chunk_text(patched[-1])

        label_from_end = extract_table_label_from_end(previous_text)

        if label_from_end:

            patched_text = (
                f"{label_from_end} (Начало)\n\n{text}"
            )

            patched.append(
                set_chunk_text(chunk, patched_text)
            )

            continue

        label_from_start = extract_table_label_from_start(previous_text)

        if label_from_start:

            clean_label = re.sub(
                r"\s+\((Начало|Продолжение)\)",
                "",
                label_from_start,
                flags=re.IGNORECASE,
            )

            patched_text = (
                f"{clean_label} (Продолжение)\n\n{text}"
            )

            patched.append(
                set_chunk_text(chunk, patched_text)
            )

            continue

        patched.append(chunk)

    return patched

def generate_chunks(markdown_file):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write(markdown_file)
        tmp_path = tmp.name

    doc = (Pipeline()
        .fetch_from("file", path=tmp_path)
        .process_with("markdown", tokenizer=emb_tokenizer)
        .chunk_with("table", chunk_size=1024, tokenizer=emb_tokenizer)
        .chunk_with("recursive", chunk_size=1024, tokenizer=emb_tokenizer)
        .run()
    )

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    doc.chunks = patch_table_chunks(doc.chunks)
    return doc.chunks

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def embed_chunks(chunks, batch_size: int = 64, max_length: int = 1024) -> Tensor:
    
    device = emb_model.device
    all_embeddings = []

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            
            inputs = emb_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = emb_model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def merge_chunks(chunks, filename):
    results = []

    embeddings = embed_chunks([chunk.text for chunk in chunks])

    for i, chunk in enumerate(chunks):
        prefix = f"Документ: {filename}\nЧанк: №{i}\n\n"
        text = chunk.text
        emb = embeddings[i].cpu().tolist()

        results.append({
            "doc_id": filename,
            "chunk_id": i,
            "content": prefix+text,
            "embedding": emb
        })

    return results

conn = psycopg2.connect(DB_CONN)
cur = conn.cursor()

def save_to_db(chunks, path, is_ocr):
    meta = {
            "path": str(path),
            "add_time": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "is_ocr": is_ocr
        }
    
    params = [
        (
            row["doc_id"],
            row["chunk_id"],
            row["content"],
            row["embedding"]
        )
            for row in chunks
    ]

    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO demo_data (doc_id, chunk_id, content, embedding) VALUES (%s, %s, %s, %s)""", params)
        cur.execute("""INSERT INTO demo_data_metadata (doc_id, metadata) VALUES (%s, %s::jsonb)""", (chunks[0]["doc_id"], json.dumps(meta)))
        conn.commit()

def load_docs(folder_path, use_ocr=False, picture_desc=False):
    cur.execute("SELECT DISTINCT doc_id FROM demo_data")
    unique_docs = cur.fetchall()
    unique = set()
    for value in unique_docs:
        unique.add(value[0])

    word_converter, main_converter = get_converters(use_ocr=use_ocr, picture_desc=picture_desc)

    for filename in tqdm(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        if 'txt' in ext:
            file = Path(path)
            file.rename(file.with_suffix('.md'))
            filename = filename.replace('.txt', '.md')
            path = os.path.join(folder_path, filename)

        if filename in unique:
            print(f"Документ {filename} уже обработан.")
            continue

        try:
            print(f"Перевод в md... {filename}")
            if "docx" in ext.lower():
                result = word_converter.convert(path)
            else:
                result = main_converter.convert(path)

            doc = result.document.export_to_markdown()

            if 'pdf' in ext.lower() and len(doc) < 250 and use_ocr==False:
                print(f"Файл оставлен для OCR. {filename}")
                continue

        except Exception as e:
            print(f"Формат файла {filename} не поддерживается. {e}")
            continue
        
        print(f"Нарезка чанков... {filename}")
        chunks = generate_chunks(doc)
        print(f"\rГенерация эмбеддингов... {filename}")
        results = merge_chunks(chunks, filename)
 
        #temp_path = os.path.join("/wrk/data/База данных pdf", filename)

        save_to_db(results, path, use_ocr)
        
        with open("/wrk/result.txt", "a", encoding="utf-8") as f:
            f.write(f"{filename} сохранен\n")

        torch.cuda.empty_cache()
        
        print(f"{filename}: {len(chunks)} чанков сохранено. {datetime.now().strftime('%d.%m.%Y %H:%M')}")

load_docs(FOLDER_PATH, use_ocr=False, picture_desc=True)
# После прохода по всей директории запускаем OCR для документов, которые не обработались:
# load_docs(FOLDER_PATH, use_ocr=True, picture_desc=True)

# После загрузки документов проиндексируйте чанки с помощью команды ниже.  
# В случае, если вы будете добавлять новые документы, не забывайте сбрасывать индексы и создавать их заново (для индексации всех докуметов)  

# CREATE INDEX ON demo_data USING ivfflat (embedding vector_cosine_ops);  
#     --NOTICE:  ivfflat index created with little data  
#     --DETAIL:  This will cause low recall.  
#     --HINT:  Drop the index until the table has more data.  
#     (DROP INDEX IF EXISTS name)

# Затем необходимо добавить колонку и индексы для алгоритма BM25:  
# ALTER TABLE demo_data ADD COLUMN tsv tsvector;  
# UPDATE demo_data SET tsv = to_tsvector('russian', content);  
# CREATE INDEX idx_demo_data_tsv ON demo_data USING gin(tsv);  

# А также необходимо создать все таблицы для будущего data layer:  
# CREATE TABLE users (
#     "id" UUID PRIMARY KEY,
#     "identifier" TEXT NOT NULL UNIQUE,
#     "metadata" JSONB NOT NULL,
#     "createdAt" TEXT
# );

# CREATE TABLE IF NOT EXISTS threads (
#     "id" UUID PRIMARY KEY,
#     "createdAt" TEXT,
#     "name" TEXT,
#     "userId" UUID,
#     "userIdentifier" TEXT,
#     "tags" TEXT[],
#     "metadata" JSONB,
#     FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
# );

# CREATE TABLE IF NOT EXISTS steps (
#     "id" UUID PRIMARY KEY,
#     "name" TEXT NOT NULL,
#     "type" TEXT NOT NULL,
#     "threadId" UUID NOT NULL,
#     "parentId" UUID,
#     "streaming" BOOLEAN NOT NULL,
#     "waitForAnswer" BOOLEAN,
#     "isError" BOOLEAN,
#     "metadata" JSONB,
#     "tags" TEXT[],
#     "input" TEXT,
#     "output" TEXT,
#     "createdAt" TEXT,
#     "command" TEXT,
#     "start" TEXT,
#     "end" TEXT,
#     "generation" JSONB,
#     "showInput" TEXT,
#     "language" TEXT,
#     "indent" INT,
#     "defaultOpen" BOOLEAN,
#     FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
# );

# CREATE TABLE IF NOT EXISTS elements (
#     "id" UUID PRIMARY KEY,
#     "threadId" UUID,
#     "type" TEXT,
#     "url" TEXT,
#     "chainlitKey" TEXT,
#     "name" TEXT NOT NULL,
#     "display" TEXT,
#     "objectKey" TEXT,
#     "size" TEXT,
#     "page" INT,
#     "language" TEXT,
#     "forId" UUID,
#     "mime" TEXT,
#     "props" JSONB,
#     FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
# );

# CREATE TABLE IF NOT EXISTS feedbacks (
#     "id" UUID PRIMARY KEY,
#     "forId" UUID NOT NULL,
#     "threadId" UUID NOT NULL,
#     "value" INT NOT NULL,
#     "comment" TEXT,
#     FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
# );