import os
import uuid
import openai
import chromadb
from dotenv import load_dotenv

# 🔐 환경변수에서 API 키 로딩
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📌 Chroma 클라이언트 초기화 (항상 프로젝트 루트 기준)
CHROMA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../chroma_db"))
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="schema", embedding_function=None)

# 🧠 GPT 임베딩 함수
def gpt_embed(text: str) -> list:
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ GPT 임베딩 실패: {e}")
        return None

# 🔼 업서트 함수 (GPT 임베딩 사용)
def upsert_schema(text_id: str, text: str):
    embedding = gpt_embed(text)
    if embedding is None:
        print(f"⚠️ 업서트 스킵 (임베딩 실패): {text}")
        return

    collection.upsert(
        documents=[text],
        ids=[text_id],
        embeddings=[embedding],
        metadatas=[{"source": "manual"}]
    )
    print(f"✅ 업서트 완료: {text_id}")

# 🔍 검색 함수 (GPT 임베딩 사용)
def search_similar_schema(query: str, top_k: int = 3) -> list:
    embedding = gpt_embed(query)
    if embedding is None:
        print("❌ 쿼리 임베딩 실패")
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    if results and results.get("documents") and results["documents"][0]:
        return results["documents"][0]

    print("🔍 검색 결과 없음")
    return []

# 🧪 테스트 (직접 실행할 때만)
if __name__ == "__main__":
    print("✅ 현재 문서 수:", collection.count())

    # 테스트 업서트
    upsert_schema(str(uuid.uuid4()), "작업자 이름과 공정명을 조회합니다.")
    upsert_schema(str(uuid.uuid4()), "설비 상태를 확인합니다.")
    upsert_schema(str(uuid.uuid4()), "작업 지시서 생성일자를 가져옵니다.")

    # 테스트 검색
    results = search_similar_schema("공정명이랑 작업자 알려줘")
    print("🔍 검색 결과:", results)
