from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from openai import OpenAI
import os

# 1. Milvus 연결
connections.connect("default", host="localhost", port="19530")

# 2. OpenAI 클라이언트 설정 (🔑 API 키 넣어줘)
openai = OpenAI(api_key="")


# 3. 컬렉션 이름
collection_name = "sql_fewshot_collection"

# 4. 기존 컬렉션이 있으면 삭제
if utility.has_collection(collection_name):
    Collection(name=collection_name).drop()
    print(f"🗑 기존 컬렉션 '{collection_name}' 삭제 완료")

# 5. 새 컬렉션 생성
fields = [
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512, is_primary=True, auto_id=False),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)
]
schema = CollectionSchema(fields=fields, description="Fewshot SQL examples")
collection = Collection(name=collection_name, schema=schema)
# collection.load()

# 6. 예시 데이터
examples = [
    {
        "question": "5월 12일에 숭의 국사에서 승인한 승인자 목록?",
        "answer": "SELECT DISTINCT APVR_ID\nFROM WFM_WRKPRMSSTTUS_BAS\nWHERE TO_CHAR(WRK_PRMS_RQT_DATE, 'YYYY-MM-DD') = '2024-05-12'\n  AND WRK_OBDNG_ID = '숭의'\n  AND WRK_PRMS_RQT_STTUS_CD = '2';"
    },
    {
        "question": "2024년 5월 26일부터 5월 28일까지 전체 국사에서 버켓차량 오더 개수는?",
        "answer": "SELECT COUNT(*)\nFROM WFM_WRKPRMSSTTUS_BAS\nWHERE WRK_PRMS_RQT_DATE BETWEEN TO_DATE('2024-05-26', 'YYYY-MM-DD') AND TO_DATE('2024-05-28', 'YYYY-MM-DD')\n  AND RISK_TRT_TYPE_CD = '1';"
    },
    {
        "question": "서초국사에서 승인완료가 아닌 오더 정보는?",
        "answer": "SELECT *\nFROM WFM_WRKPRMSSTTUS_BAS\nWHERE WRK_OBDNG_ID = '서초'\n  AND WRK_PRMS_RQT_STTUS_CD != '2';"
    }
]

# 7. 임베딩 생성 함수
def embed(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# 8. 삽입할 데이터 준비
questions = []
answers = []
embeddings = []

for ex in examples:
    questions.append(ex["question"])
    answers.append(ex["answer"])
    embeddings.append(embed(ex["question"]))

# 9. 삽입
collection.insert([questions, answers, embeddings])
print("✅ 새 컬렉션 생성 및 데이터 삽입 완료")