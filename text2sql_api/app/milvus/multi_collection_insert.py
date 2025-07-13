
import os
import uuid
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


# 📐 Define schema for biz_term_collection
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

schema = CollectionSchema(fields)
if utility.has_collection("biz_term_collection"):
    Collection("biz_term_collection").drop()
collection = Collection(name="biz_term_collection", schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("📦 Created collection: biz_term_collection")

file_path = os.path.join("data", "biz_term_sentences.txt")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    text = line.strip()
    if not text:
        continue

    doc_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    collection.insert([
        [doc_id],
        [text],
        [text],
        [embedding]
    ])

collection.flush()
print("✅ Inserted data into: biz_term_collection")


# 📐 Define schema for column_info_collection
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="column_info", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

schema = CollectionSchema(fields)
if utility.has_collection("column_info_collection"):
    Collection("column_info_collection").drop()
collection = Collection(name="column_info_collection", schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("📦 Created collection: column_info_collection")

file_path = os.path.join("data", "column_info_sentences.txt")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    text = line.strip()
    if not text:
        continue

    doc_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    collection.insert([
        [doc_id],
        [text],
        [text],
        [embedding]
    ])

collection.flush()
print("✅ Inserted data into: column_info_collection")


# 📐 Define schema for custom_sql_collection
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

schema = CollectionSchema(fields)
if utility.has_collection("custom_sql_collection"):
    Collection("custom_sql_collection").drop()
collection = Collection(name="custom_sql_collection", schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("📦 Created collection: custom_sql_collection")

file_path = os.path.join("data", "custom_fewshot_sql_sentences.txt")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    text = line.strip()
    if not text:
        continue

    doc_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    collection.insert([
        [doc_id],
        [text],
        [text],
        [embedding]
    ])

collection.flush()
print("✅ Inserted data into: custom_sql_collection")


# 📐 Define schema for ddl_collection
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="raw_ddl", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

schema = CollectionSchema(fields)
if utility.has_collection("ddl_collection"):
    Collection("ddl_collection").drop()
collection = Collection(name="ddl_collection", schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("📦 Created collection: ddl_collection")

file_path = os.path.join("data", "ddl_sentences.txt")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    text = line.strip()
    if not text:
        continue

    doc_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    collection.insert([
        [doc_id],
        [text],
        [text],
        [embedding]
    ])

collection.flush()
print("✅ Inserted data into: ddl_collection")


# 📐 Define schema for table_meta_collection
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="table_meta", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

schema = CollectionSchema(fields)
if utility.has_collection("table_meta_collection"):
    Collection("table_meta_collection").drop()
collection = Collection(name="table_meta_collection", schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("📦 Created collection: table_meta_collection")

file_path = os.path.join("data", "table_meta_sentences.txt")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    text = line.strip()
    if not text:
        continue

    doc_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    collection.insert([
        [doc_id],
        [text],
        [text],
        [embedding]
    ])

collection.flush()
print("✅ Inserted data into: table_meta_collection")
