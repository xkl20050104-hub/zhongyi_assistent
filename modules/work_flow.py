from langsmith import traceable
from llama_index.core import PromptTemplate, Settings

@traceable(run_type="retriever", name="Retrieve_Documents")
def retrieve_docs(index, question):
    retriever = index.as_retriever(similarity_top_k=3)
    return retriever.retrieve(question)

@traceable(run_type="chain", name="Format_Prompt")
def format_prompt(question, nodes):
    context_str = "\n\n".join([n.get_content() for n in nodes])
    qa_prompt_tmpl = PromptTemplate(
        "上下文信息如下：\n{context_str}\n请根据上述内容回答：{query_str}\n回答："
    )
    return qa_prompt_tmpl.format(context_str=context_str, query_str=question)

@traceable(run_type="llm", name="LLM_Generation")
def call_llm(prompt):
    # 这里是 Token 消耗产生的核心点
    response = Settings.llm.complete(prompt)
    return response

@traceable(run_type="chain", name="TCM_Full_RAG_Flow")
def run_tcm_rag(index, question):
    # 1. 检索
    nodes = retrieve_docs(index, question)
    # 2. 拼接
    full_prompt = format_prompt(question, nodes)
    # 3. 生成
    response = call_llm(full_prompt)
    return response