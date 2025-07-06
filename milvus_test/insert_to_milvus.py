import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import openai

# ✅ 사용자 설정
openai.api_key = ""
data_dir = r"C:\\Users\\Kunyeol Na\\Desktop\\mochi_query\\data"
file_to_collection = {
    "s_biz_term_sentences.txt": "biz_term_collection",
    "s_column_info_sentences.txt": "column_meta_collection",
    "s_custom_fewshot_sql_sentences.txt": "sql_fewshot_collection",
    "s_ddl_sentences.txt": "ddl_collection",
    "s_table_meta_sentences.txt": "table_meta_collection",
}

# ✅ Milvus 연결 (Docker에서 실행 중인 기본 포트)
connections.connect("default", host="localhost", port="19530")

# ✅ 임베딩 함수 (OpenAI)
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# ✅ Milvus에 컬렉션 생성 및 벡터 insert
def insert_to_milvus(collection_name, texts):
    # 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
    ]
    schema = CollectionSchema(fields, description=f"Collection for {collection_name}")

    # 기존 컬렉션 있으면 삭제 후 새로 생성
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
    col = Collection(name=collection_name, schema=schema)

    # 데이터 준비 및 insert
    data = [[], []]  # text, embedding
    for text in texts:
        embedding = get_embedding(text)
        data[0].append(text)
        data[1].append(embedding)

    col.insert([data[0], data[1]])
    print(f"✅ Inserted {len(texts)} records into '{collection_name}'")

# ✅ 전체 실행 루프
for filename, collection_name in file_to_collection.items():
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {filename}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    insert_to_milvus(collection_name, lines)

print("🎉 모든 데이터 삽입 완료!")
