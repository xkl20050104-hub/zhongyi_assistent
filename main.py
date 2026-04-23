import os
from dotenv import load_dotenv
from loguru import logger
from modules.data_clean import clean_tcm_text
from modules.query_engine import get_query_engine

load_dotenv()


def run():
    # 路径使用 r 前缀规避编码报错
    raw_doc = r"data/《中医临床诊疗术语第2部分：证候》（修订版）.docx"
    clean_txt = r"data/cleaned_corpus.txt"
    storage_path = r"data/storage"

    logger.info("=== 中医诊疗助手启动中 ===")

    # 1. 自动执行数据清洗
    if not os.path.exists(clean_txt):
        logger.info("检测到未处理语料，启动清洗模块...")
        clean_tcm_text(raw_doc, clean_txt)
    else:
        logger.debug("清洗后的语料已存在，跳过清洗步骤")

    # 2. 获取引擎
    engine = get_query_engine(data_dir="data", persist_dir=storage_path)

    # 3. 交互循环
    print("\n" + "=" * 30)
    print("  中医诊疗助手 (输入 q 退出)")
    print("=" * 30)

    while True:
        question = input("\n问：")
        if question.lower() == 'q':
            logger.info("用户主动退出程序")
            break

        logger.info(f"收到用户提问: {question}")
        response = engine.query(question)
        print(f"\n答：{response}")


if __name__ == "__main__":
    run()