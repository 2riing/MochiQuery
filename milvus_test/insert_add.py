import json
import openai
from pymilvus import Collection, connections, DataType, FieldSchema, CollectionSchema, utility
from tqdm import tqdm

# OpenAI API key 설정
openai.api_key = """"

# Milvus 연결
connections.connect(alias="default", host="localhost", port="19530")

# 컬렉션 이름
collection_name = "ddl_collection"

# 기존 컬렉션이 있다면 삭제
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
    print(f"🗑️ 기존 컬렉션 '{collection_name}' 삭제 완료")

# 새 컬렉션 정의
fields = [
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False),
    FieldSchema(name="columns", dtype=DataType.JSON),
    FieldSchema(name="primary_key", dtype=DataType.JSON),
    FieldSchema(name="raw_ddl", dtype=DataType.VARCHAR, max_length=30000),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]
schema = CollectionSchema(fields, description="DDL 구조화 임베딩")
collection = Collection(name=collection_name, schema=schema)
print(f"📦 새 컬렉션 '{collection_name}' 생성 완료")

# JSON 파일 로드
with open("data/parsed_ddl_records.json", "r", encoding="utf-8") as f:
    ddl_data = json.load(f)

# 임베딩 함수
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# 임베딩 및 insert
texts = []
embeddings = []
table_names = []
columns = []
primary_keys = []
raw_ddls = []

for record in tqdm(ddl_data):
    table_names.append(record["table_name"])
    columns.append(record["columns"])
    primary_keys.append(record["primary_key"])
    raw_ddls.append(record["raw_ddl"])
    
    text = json.dumps(record, ensure_ascii=False)
    texts.append(text)
    embeddings.append(get_embedding(text))

# Milvus에 insert
collection.insert([table_names, columns, primary_keys, raw_ddls, texts, embeddings])
collection.flush()
print("✅ Insert complete.")
