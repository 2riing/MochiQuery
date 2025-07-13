import os
import uuid
import openai
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm
from dotenv import load_dotenv

# 🔐 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Milvus 연결 (Docker에서 실행 중인 기본 포트)
connections.connect("default", host="localhost", port="19530")
collection_name = "semantic_text_collection"

# 🧼 기존 삭제
# if utility.has_collection(collection_name):
#     Collection(collection_name).drop()
#     print("🗑️ 기존 컬렉션 삭제")

# 📐 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64), # 파일 출처
    # 공통 또는 조건부 필드
    FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="column_info", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="raw_ddl", dtype=DataType.VARCHAR, max_length=30000),
    FieldSchema(name="table_meta", dtype=DataType.VARCHAR, max_length=2048),
    # 검색용 텍스트 & 벡터
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]

# 컬렉션 생성
schema = CollectionSchema(fields)
collection = Collection(name=collection_name, schema=schema)

# embedding 필드에 인덱스 생성
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print(f"📦 컬렉션 '{collection_name}' 생성 완료")

# 📌 임베딩 함수
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# 📁 파일 목록 및 매핑
data_path = "../data"
file_map = {
    "biz_term_sentences.txt": ("term", "biz_term"),
    "column_info_sentences.txt": ("column_info", "column_info"),
    "custom_fewshot_sql_sentences.txt": ("question", "custom_sql"),
    "ddl_sentences.txt": ("raw_ddl", "ddl"),
    "table_meta_sentences.txt": ("table_meta", "table_meta"),
}

# 📥 데이터 읽고 insert
for filename, (target_field, source_name) in file_map.items():
    file_path = os.path.join(data_path, filename)
    print(f"📄 Processing: {filename}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        text = line.strip()
        if not text:
            continue

        doc_id = str(uuid.uuid4())
        embedding = get_embedding(text)

        # 모든 필드는 기본값 "" 또는 None
        record = {
            "id": doc_id,
            "source": source_name,
            "term": "",
            "column_info": "",
            "question": "",
            "raw_ddl": "",
            "table_meta": "",
            "text": text,
            "embedding": embedding,
        }
        record[target_field] = text

        # 순서대로 insert
        collection.insert([
            [record["id"]],
            [record["source"]],
            [record["term"]],
            [record["column_info"]],
            [record["question"]],
            [record["raw_ddl"]],
            [record["table_meta"]],
            [record["text"]],
            [record["embedding"]],
        ])
        
    collection.flush()

print("✅ 모든 파일 처리 완료")
