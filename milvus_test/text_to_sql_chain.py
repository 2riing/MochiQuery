from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from milvus_search_tool import search_and_return_text

def get_prompt(user_question: str) -> str:
    # term_info = search_and_return_text("biz_term_collection", user_question)
    # column_info = search_and_return_text("column_meta_collection", user_question)
    # table_info = search_and_return_text("table_meta_collection", user_question)

    term_info = ""
    column_info = ""
    table_info = ""

    context = f"""[용어 설명]\n{term_info}\n\n[컬럼 설명]\n{column_info}\n\n[테이블 정보]\n{table_info}"""
    return f"{context}\n\n위 정보를 참고하여 다음 자연어 질문에 대한 SQL을 작성하세요:\n{user_question}"

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = PromptTemplate.from_template("{full_prompt}")

chain = LLMChain(llm=llm, prompt=prompt)  