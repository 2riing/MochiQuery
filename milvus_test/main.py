from fastapi import FastAPI
from langserve import add_routes
from text_to_sql_chain import chain, get_prompt

from langchain.schema.runnable import RunnableLambda

app = FastAPI()

# 자연어를 SQL로 바꾸는 체인을 래핑
def run_text_to_sql(input: dict) -> dict:
    question = input["question"]
    full_prompt = get_prompt(question)  # 필요 없다면 그냥 question 사용
    return chain.invoke({"full_prompt": full_prompt})

text_to_sql_chain = RunnableLambda(run_text_to_sql)

add_routes(app, chain, path="/text-to-sql")