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

collection = Collection(name="semantic_text_collection")
collection.load()

def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


# 🔍 검색 함수
def search_similar(query_text, top_k=5):
    query_embedding = get_embedding(query_text)
    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["source", "text"]
    )

    print(f"\n🔎 검색 결과 (Query: '{query_text}')")
    for i, hit in enumerate(results[0], 1):
        print(f"{i}. [{hit.entity.source}] {hit.entity.text[:100]}... (score: {hit.distance:.4f})")

# 🔍 테스트 실행
search_similar("고객 작업 접수번호가 뭐야?")
search_similar("DDL 정의 예시 알려줘")
