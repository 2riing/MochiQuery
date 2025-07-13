import os
import openai
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv

# 🔐 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Milvus 연결 (Docker에서 실행 중인 기본 포트)
connections.connect("default", host="localhost", port="19530")

# collection load 
# collection = Collection(name="semantic_text_collection")
# collection.load()

# 임베딩 
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


# 🔍 Milvus 벡터 검색 함수
def search_similar_schema(query: str, top_k: int = 3, collection_name: str = "semantic_text_collection") -> list:
    if not utility.has_collection(collection_name):
        print(f"❌ 컬렉션 '{collection_name}'이 존재하지 않습니다.")
        return []

    embedding = get_embedding(query)
    if embedding is None:
        return []

    try:
        collection = Collection(name=collection_name)
        collection.load()

        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["source", "text"]
        )

        return [
            {
                "source": hit.entity.source,
                "text": hit.entity.text,
                "score": hit.distance
            }
            for hit in results[0]
        ]

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return []
    
    
# Table검색  > 그외 검색 
def chained_table_first_search(query: str, top_k: int = 3) -> list:
    from pymilvus import Collection, utility

    def _search(collection_name, embedding, output_fields):
        if not utility.has_collection(collection_name):
            print(f"❌ 컬렉션 '{collection_name}'이 존재하지 않습니다.")
            return []
        try:
            col = Collection(name=collection_name)
            col.load()
            results = col.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=output_fields
            )
            return results[0]
        except Exception as e:
            print(f"❌ {collection_name} 검색 실패: {e}")
            return []

    # 1단계: table_meta_collection에서 테이블 이름 추출
    embedding = get_embedding(query)
    table_meta_hits = _search("table_meta_collection", embedding, ["table_meta", "text"])
    table_names = [hit.entity.table_meta for hit in table_meta_hits]

    combined_results = []

    # 2단계: 각 table_name을 가지고 나머지 컬렉션에서 검색
    for table_name in table_names:
        # 각 컬렉션마다 검색할 필드 정의
        search_fields = {
            "column_info_collection": ["table_name", "column_name", "description"],
            "custom_sql_collection": ["table_name", "fewshot_answer"],
            "ddl_collection": ["table_name", "columns", "primary_key"]
        }
        for col_name, fields in search_fields.items():
            sub_results = _search(col_name, get_embedding(table_name), fields)
            for hit in sub_results:
                combined_results.append({
                    "table_name": table_name,
                    "collection": col_name,
                    "score": hit.distance,
                    **{field: getattr(hit.entity, field, "") for field in fields}
                })

    return combined_results