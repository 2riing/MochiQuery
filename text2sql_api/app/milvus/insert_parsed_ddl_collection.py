
import os
import uuid
import json
import openai
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm
from dotenv import load_dotenv

# 🔐 Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 📌 Embedding function
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# 📐 Schema for ddl_collection
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="columns", dtype=DataType.VARCHAR, max_length=30000),
    FieldSchema(name="primary_key", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="raw_ddl", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

schema = CollectionSchema(fields)
collection_name = "ddl_collection"
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
collection = Collection(name=collection_name, schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("📦 Created collection: ddl_collection")

# 📁 Load JSON DDL records
with open("../data/parsed_ddl_records.json", "r", encoding="utf-8") as f:
    ddl_records = json.load(f)

for record in tqdm(ddl_records):
    table_name = record.get("table_name", "")
    columns = json.dumps(record.get("columns", []), ensure_ascii=False)
    primary_key = ", ".join(record.get("primary_key", []))
    raw_ddl = record.get("raw_ddl", "")

    # 검색용 텍스트: 테이블명 + 컬럼 + PK + 일부 DDL
    text = f"{table_name} 테이블: {primary_key} PK 포함. 컬럼 정의: {columns[:2000]}..."

    doc_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    collection.insert([
        [doc_id],
        [table_name],
        [columns],
        [primary_key],
        [raw_ddl],
        [text],
        [embedding]
    ])

collection.flush()
print("✅ Inserted parsed DDLs into: ddl_collection")
