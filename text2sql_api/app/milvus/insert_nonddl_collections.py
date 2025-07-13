
import os
import uuid
import openai
import re
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


# 📦 Processing: biz_term_collection
# fields = [FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True), FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)]
# schema = CollectionSchema(fields)
# if utility.has_collection("biz_term_collection"):
#     Collection("biz_term_collection").drop()
# collection = Collection(name="biz_term_collection", schema=schema)

# collection.create_index(
#     field_name="embedding",
#     index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
# )

# file_path = os.path.join("../data", "biz_term_sentences.txt")
# with open(file_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# for line in tqdm(lines):
#     text = line.strip()
#     if not text:
#         continue

#     record = {}
#     record["id"] = str(uuid.uuid4())
#     record["text"] = text
#     record["embedding"] = get_embedding(text)
#     record["term"] = text
    
#     collection.insert([
#         [record["id"]],
#         [record["term"]],
#         [record["text"]],
#         [record["embedding"]]
#     ])

# collection.flush()
# print("✅ Finished: biz_term_collection")


# # 📦 Processing: column_info_collection
# fields = [FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True), FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=512), FieldSchema(name="column_name", dtype=DataType.VARCHAR, max_length=512), FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)]
# schema = CollectionSchema(fields)
# if utility.has_collection("column_info_collection"):
#     Collection("column_info_collection").drop()
# collection = Collection(name="column_info_collection", schema=schema)

# collection.create_index(
#     field_name="embedding",
#     index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
# )

# file_path = os.path.join("../data", "column_info_sentences.txt")
# with open(file_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# for line in tqdm(lines):
#     text = line.strip()
#     if not text:
#         continue

#     record = {}
#     record["id"] = str(uuid.uuid4())
#     record["text"] = text
#     record["embedding"] = get_embedding(text)


#     m = re.match(r"\[(.*?)\] 테이블의 \[(.*?)\] 컬럼은 (.*)", text)
#     if m:
#         record["table_name"] = m.group(1)
#         record["column_name"] = m.group(2)
#         record["description"] = m.group(3)
        

#     collection.insert([
#         [record["id"]],
#         [record.get("table_name", "")],
#         [record.get("column_name", "")],
#         [record.get("description", "")],
#         [record["text"]],
#         [record["embedding"]]
#     ])

# collection.flush()
# print("✅ Finished: column_info_collection")


# # 📦 Processing: custom_sql_collection
# fields = [FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True), FieldSchema(name="fewshot_question", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="fewshot_answer", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)]
# schema = CollectionSchema(fields)
# if utility.has_collection("custom_sql_collection"):
#     Collection("custom_sql_collection").drop()
# collection = Collection(name="custom_sql_collection", schema=schema)

# collection.create_index(
#     field_name="embedding",
#     index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
# )

# file_path = os.path.join("../data", "custom_fewshot_sql_sentences.txt")
# with open(file_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# for line in tqdm(lines):
#     text = line.strip()
#     if not text:
#         continue

#     record = {}
#     record["id"] = str(uuid.uuid4())
#     record["text"] = text
#     record["embedding"] = get_embedding(text)


#     if "질문:" in text and "SQL:" in text:
#         parts = text.split("SQL:")
#         record["fewshot_question"] = parts[0].replace("질문:", "").strip()
#         record["fewshot_answer"] = parts[1].strip()
        

#     collection.insert([
#         [record["id"]],
#         [record.get("fewshot_question", "")],
#         [record.get("fewshot_answer", "")],
#         [record["text"]],
#         [record["embedding"]]
#     ])

# collection.flush()
# print("✅ Finished: custom_sql_collection")


# 📦 Processing: table_meta_collection
fields = [FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True), FieldSchema(name="table_meta", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768), FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)]
schema = CollectionSchema(fields)
if utility.has_collection("table_meta_collection"):
    Collection("table_meta_collection").drop()
collection = Collection(name="table_meta_collection", schema=schema)

collection.create_index(
    field_name="embedding",
    index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
)

file_path = os.path.join("../data", "table_meta_sentences.txt")
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    text = line.strip()
    if not text:
        continue

    record = {}
    record["id"] = str(uuid.uuid4())
    record["text"] = text
    record["embedding"] = get_embedding(text)
    record["table_meta"] = text
        

    collection.insert([
        [record["id"]],
        [record["table_meta"]],
        [record["text"]],
        [record["embedding"]]
    ])


collection.flush()
print("✅ Finished: table_meta_collection")
