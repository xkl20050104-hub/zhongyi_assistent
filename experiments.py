import os
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langsmith import evaluate
from modules.work_flow import run_tcm_rag
from modules.query_engine import get_index

# 1. 环境准备
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化索引
index = get_index(data_dir="data", persist_dir="data/storage")


# 2. 定义结构化模型（增加容错性）
class SimilarityScore(BaseModel):
    # 使用 float 接收分数，因为模型常返回 1.0 或 0.85
    similarity_score: float = Field(description="0到10之间的语义相似度分数")
    # 设为 Optional 并给默认值，防止模型漏写 reasoning 字段时崩溃
    reasoning: Optional[str] = Field(default="模型未返回具体理由", description="理由")


# 3. 构建评估器函数
def tcm_semantic_evaluator(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
    # 统一使用你日志里确认的键名 'question'
    input_question = inputs.get("question")
    # 这里的键名请根据你网页端的显示灵活切换，通常 LangSmith 会映射为 'answer'
    reference_res = reference_outputs.get("answer") or reference_outputs.get("Reference Outputs")
    actual_res = outputs.get("output")

    if not actual_res or not reference_res:
        return {"score": 0, "key": "semantic_similarity", "comment": "数据读取为空"}

    try:
        # 使用 parse 方法进行结构化输出
        completion = client.beta.chat.completions.parse(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位中医专家评估员。请对比参考答案和系统输出的语义一致性。\n"
                        "必须严格按 JSON 格式输出，包含以下字段：\n"
                        "- similarity_score: 数字 (0-10)\n"
                        "- reasoning: 理由字符串\n"
                        "即使回答不完美，也必须返回这两个字段。"
                    )
                },
                {"role": "user", "content": f"问题: {input_question}\n参考: {reference_res}\n系统: {actual_res}"}
            ],
            response_format=SimilarityScore,
            temperature=0.1,  # 降低随机性，让模型更稳定
        )

        result = completion.choices[0].message.parsed

        # 转换分数：LangSmith 界面更喜欢 0-1 之间的值
        # 如果模型给了 0-10，我们就除以 10
        raw_score = result.similarity_score
        final_score = raw_score / 10.0 if raw_score > 1 else raw_score

        return {
            "score": final_score,
            "key": "semantic_similarity",
            "comment": result.reasoning
        }
    except Exception as e:
        # 极其重要：即使 LLM 报错，也返回一个 0 分反馈，确保 LangSmith 界面不报错
        print(f"评估器内部报错: {str(e)}")
        return {"score": 0, "key": "semantic_similarity", "comment": f"评估异常: {str(e)}"}


# 4. 定义实验目标函数
def tcm_target_function(inputs: dict):
    question = inputs.get("question")
    if not question:
        return {"output": "无输入"}
    response = run_tcm_rag(index, question)
    return {"output": str(response)}


# 5. 执行评估实验
if __name__ == "__main__":
    dataset_name = "中医语料"
    print(f"正在启动评估流程...")

    evaluate(
        tcm_target_function,
        data=dataset_name,
        evaluators=[tcm_semantic_evaluator],
        experiment_prefix="TCM_RAG_Final_Test",
        metadata={
            "judge_model": "qwen-max",  # 负责打分的模型
            "rag_model": "qwen-plus",  # 负责生成回答的模型
            "version": "1.0",
            "model": "3"  # 这里的 key 必须和网页端列名对应
        }
    )
    print("评估完成！请刷新 LangSmith 网页查看指标。")