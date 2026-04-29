import os
from dotenv import load_dotenv

load_dotenv()

from loguru import logger
from modules.data_clean import clean_tcm_text
from modules.query_engine import get_index
from modules.work_flow import run_tcm_rag

def run():
    raw_doc = r"data/《中医临床诊疗术语第2部分：证候》（修订版）.docx"
    clean_txt = r"data/cleaned_corpus.txt"
    storage_path = r"data/storage"

    logger.info("=== 中医诊疗助手启动中 ===")

    if not os.path.exists(clean_txt):
        logger.info("启动语料清洗...")
        clean_tcm_text(raw_doc, clean_txt)

    index = get_index(data_dir="data", persist_dir=storage_path)

    print("\n" + "=" * 30)
    print("  中医诊疗助手 (输入 q 退出)")
    print("=" * 30)

    while True:
        question = input("\n问：")
        if question.lower() == 'q':
            logger.info("程序结束")
            break

        if not question.strip(): continue

        response = run_tcm_rag(index, question)
        print(f"\n答：{response}")

if __name__ == "__main__":
    run()
