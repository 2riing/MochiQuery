from pymilvus import Collection, connections
import os
import uuid
import openai
import chromadb
from dotenv import load_dotenv

# 🔐 환경변수에서 API 키 로딩
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Milvus 연결
connections.connect("default", host="localhost", port="19530")

# ✅ 임베딩 함수 (insert 코드와 동일해야 함)
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def get_output_fields(collection_name):
    if collection_name == "sql_fewshot_collection":
        return ["answer"]
    elif collection_name == "ddl_collection":
        return ["raw_ddl"]
    else:
        return ["text"]

# ✅ 검색 함수
def search_in_milvus(collection_name, query_text, top_k=3):
    query_embedding = get_embedding(query_text)
    col = Collection(collection_name)
    col.load()

    search_params = {
    "metric_type": "COSINE",
    "params": {
            "ef": 64,         # HNSW 전용
            "nprobe": 16      # IVF 계열 전용
        }
    }

    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=get_output_fields(collection_name)
    )

    for i, hit in enumerate(results[0]):
        print(f"\n🔹 TOP {i+1}")
        print(query_text)
        print(f"Score: {hit.distance:.4f}")
        output_field = get_output_fields(collection_name)[0]
        print(f"Text: {hit.entity.get(output_field)}")
        print("---")

# ✅ 테스트 실행
if __name__ == "__main__":
    print("👉 Milvus 검색 테스트 중...")
    search_in_milvus("biz_term_collection", "작업접수번호는?", top_k=1)
    search_in_milvus("column_meta_collection", "신호수 작업자를 나타내는 컬럼은?", top_k=1)
    search_in_milvus("table_meta_collection", "이력정보를 저장하는 테이블?", top_k=1)
    
    search_in_milvus("sql_fewshot_collection", "승인자 목록 추출하는 쿼리?", top_k=1)
    search_in_milvus("ddl_collection", "WFM_WRKPRMSTYPE_DTL 테이블 생성하는 DDL", top_k=1)
    


def search_and_return_text(collection_name, query_text, top_k=1) -> str:
    query_embedding = get_embedding(query_text)
    col = Collection(collection_name)
    col.load()

    search_params = {
        "metric_type": "COSINE",
        "params": {
            "ef": 64,
            "nprobe": 16
        }
    }

    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=get_output_fields(collection_name)
    )

    output_field = get_output_fields(collection_name)[0]
    top_texts = [hit.entity.get(output_field) for hit in results[0]]
    return "\n".join(top_texts)