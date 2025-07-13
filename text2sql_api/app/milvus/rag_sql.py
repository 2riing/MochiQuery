import os
import openai
from text2sql_api.app.milvus.search_tool import search_similar_schema
from text2sql_api.app.milvus.search_tool import chained_table_first_search

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 프롬프트 템플릿 정의
template = """
당신은 뛰어난 SQL 전문가입니다. 아래에 주어진 질문에 대해, 각기 다른 스키마 구조(schema1, schema2 등)에 맞춰 **서로 다른 SQL 쿼리**를 각각 작성해주세요.

질문:
{question}

스키마 정보:

📘 schema1
- 테이블명: {table_name_1}
- 주요 컬럼:
  - {column_name_1}: {description_1}
  - {column_name_2}: {description_2}
- 기본키: {primary_key_1}
- DDL 컬럼 정의: {columns_1}
- 참고 SQL 예시: {fewshot_answer_1}

📘 schema2
- 테이블명: {table_name_2}
- 주요 컬럼:
  - {column_name_1}: {description_1}
  - ...
- 기본키: {primary_key_2}
- DDL 컬럼 정의: {columns_2}
- 참고 SQL 예시: {fewshot_answer_2}

요청:
- schema1을 기준으로 작성한 SQL 쿼리
- schema2를 기준으로 작성한 SQL 쿼리
- 질문에 따라 서로 구조가 달라질 수 있으므로, 각 스키마의 테이블과 컬럼 정의에 맞게 SQL을 각각 작성하세요.

SQL 쿼리:

▶ schema1 기반:
SELECT ...

▶ schema2 기반:
SELECT ...
"""
# prompt = PromptTemplate.from_template(template)


def generate_pretty_prompt_from_both(question: str, top_k: int = 2) -> str:
    schema1_results = search_similar_schema(question, top_k=top_k)
    schema2_results = chained_table_first_search(question, top_k=top_k)

    def format_schema(name: str, results: list) -> str:
        table_map = {}
        for r in results:
            tbl = r.get("table_name")
            if not tbl:
                continue
            if tbl not in table_map:
                table_map[tbl] = []
            table_map[tbl].append(r)

        if not table_map:
            return f"📘 {name}\n(검색 결과 없음)\n"

        parts = []
        for table_name, infos in table_map.items():
            lines = [f"📘 {name}", f"- 테이블명: {table_name}"]
            for info in infos:
                if info["collection"] == "column_info_collection":
                    lines.append(f"  - 컬럼: {info.get('column_name')} → {info.get('description')}")
                elif info["collection"] == "ddl_collection":
                    lines.append(f"  - PK: {info.get('primary_key')}")
                    lines.append(f"  - 컬럼 정의: {info.get('columns')}")
                elif info["collection"] == "custom_sql_collection":
                    lines.append(f"  - 참고 SQL 예시:\n    {info.get('fewshot_answer')}")
                elif "term" in info:
                    lines.append(f"  - 용어 정의: {info['term']}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    schema1_text = format_schema("schema1", schema1_results)
    print(schema1_text)
    schema2_text = format_schema("schema2", schema2_results)
    print(schema2_text)

    prompt = f"""
당신은 뛰어난 SQL 전문가입니다. 아래에 주어진 질문에 대해, 각기 다른 스키마 구조(schema1, schema2)에 맞춰 **서로 다른 SQL 쿼리**를 각각 작성해주세요.

질문:
{question}

스키마 정보:

{schema1_text}

{schema2_text}

요청:
- schema1을 기준으로 작성한 SQL 쿼리
- schema2를 기준으로 작성한 SQL 쿼리
- 질문에 따라 서로 구조가 달라질 수 있으므로, 각 스키마의 테이블과 컬럼 정의에 맞게 SQL을 각각 작성하세요.
- 조인은 PK를 참고해서 성능이 잘 나올 수 있도록 작성할 것
# 💡 질문 목적
- 아래 질문에 대해 SQL 쿼리를 작성하세요.
- 설명 없이 SQL 쿼리만 출력하세요.
- 정확하고 최적화된 SQL을 작성하는 것이 목표입니다.
- 질문과 관련된 테이블 및 컬럼만 사용하세요.
- 불필요한 조인이나 조건은 피하세요.

# 📊 데이터 처리
- 집계가 필요한 경우 SUM, COUNT, AVG, MAX, MIN을 적절히 사용하세요.
- "최근"이라는 표현은 최근 30일로 간주하고 날짜 필터를 사용하세요.
- 그룹화가 필요한 경우에만 GROUP BY를 사용하세요.
- 결과를 정렬해야 할 경우 ORDER BY를 사용하세요.
- 특별한 지시가 없다면 결과는 LIMIT 100으로 제한하세요.

# 📚 스키마 활용
- 제공된 스키마의 테이블 및 컬럼만 사용하세요.
- 사용되는 컬럼은 반드시 스키마에 존재해야 합니다.
- JOIN은 PK와 FK 관계를 기반으로 구성하세요.
- 조인 시 PK 기반으로 작성해 성능을 고려하세요.
- CROSS JOIN은 사용하지 마세요.

# 🔐 성능 최적화
- SELECT 절에는 필요한 컬럼만 명시하세요.
- WHERE 절에서는 인덱스가 걸린 컬럼을 우선 사용하세요.
- WHERE 절에서 컬럼에 함수 적용은 피하세요.
- 필터 조건은 WHERE 절 앞부분에 위치하도록 하세요.
- 가능하면 INNER JOIN을 우선 사용하세요.

# 🧠 질문 해석
- 질문의 의도를 먼저 파악한 후 SQL을 작성하세요.
- 질문이 모호할 경우 일반적인 기준으로 가정하세요.
- "상위 고객"은 매출이 높은 고객을 의미합니다.
- "활성 사용자"는 최근 30일 이내 활동한 사용자로 간주하세요.

# 🔎 JOIN 구성
- JOIN은 ON 절에 명시적 키를 사용해 연결하세요.
- 테이블 alias를 사용해 가독성을 높이세요.
- 필요한 테이블만 JOIN 하세요.
- SELECT * 는 사용하지 마세요.
- 조인 시 참조 무결성을 고려하세요.

# 🧩 쿼리 포맷
- SQL은 보기 좋게 들여쓰기 및 줄바꿈을 포함해 작성하세요.
- SQL 코드만 출력하고 해석이나 설명은 생략하세요.
- SQL 결과는 항상 ```sql 코드 블록```으로 감싸세요.
- 복잡한 조건은 서브쿼리를 사용해 표현할 수 있습니다.
- 조건 분기는 CASE WHEN 문을 사용할 수 있습니다.

# 🧪 조건 처리
- "오늘 이전"은 WHERE date < CURRENT_DATE 로 처리하세요.
- NULL 처리는 명시적으로 IS NULL 또는 IS NOT NULL로 표현하세요.
- 문자열 조건은 LIKE '%값%' 형식으로 표현하세요.
- BETWEEN이나 >=, <= 등의 비교 연산자를 정확히 사용하세요.
- 날짜 범위는 'YYYY-MM-DD' 형식으로 표현하세요.

# 📦 출력 형식
- SQL은 ```sql 블록``` 안에 작성하세요.
- 출력에는 설명 없이 SQL만 포함하세요.
- Schema1과 Schema2는 각각 분리해서 작성하세요.
- SQL 포맷은 항상 일관되게 유지하세요.
- 주석은 요청이 있을 경우에만 작성하세요.

# 🛠️ 특수 상황
- 사용 가능한 테이블이나 컬럼이 없을 경우 "해당 없음"이라고 출력하세요.
- 조인이 불가능한 경우 단일 테이블만 사용하세요.
- 집계가 명확하지 않을 경우 SUM 또는 COUNT를 기본으로 사용하세요.
- 날짜 형식이 모호한 경우 ISO 형식(YYYY-MM-DD)을 사용하세요.
- 질문이 모호하면 적절한 가정을 기반으로 작성하세요.

SQL 쿼리:

▶ schema1 기반:
SELECT ...

▶ schema2 기반:
SELECT ...
""".strip()

    return prompt


# GPT-4 호출 함수
async def generate_sql(question: str) -> str:

    # schema1 = search_similar_schema(question)
    # schema2 = chained_table_first_search(question)
    # full_prompt = prompt.format(schema1=schema1, schema2=schema2, question=question)
    full_prompt = generate_pretty_prompt_from_both(question)


    response = await openai.ChatCompletion.acreate(  # ✅ 여기 acreate!
        model="gpt-4",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0
    )

    sql = response.choices[0].message["content"]
    return sql.strip()
