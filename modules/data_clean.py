import re
import os
from docx import Document
from loguru import logger
from tqdm import tqdm


def clean_tcm_text(input_path, output_path):
    """提取 Word 文本并清洗噪声"""
    if not os.path.exists(input_path):
        logger.error(f"源文件不存在: {input_path}")
        return

    try:
        logger.info(f"开始读取文档: {input_path}")
        doc = Document(input_path)

        # 使用 tqdm 显示提取段落进度
        full_text = []
        for para in tqdm(doc.paragraphs, desc="正在提取段落"):
            if para.text.strip():
                full_text.append(para.text)

        content = '\n'.join(full_text)

        # 移除英文、斜杠
        filtered_content = re.sub('[A-Za-z/]', '', content)
        lines = filtered_content.split('\n')

        # 过滤噪声行
        cleaned_lines = []
        for line in tqdm(lines, desc="正在清洗噪声"):
            line = line.strip()
            if line and ("泛指" not in line or "一类证候" not in line):
                cleaned_lines.append(line)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))

        logger.success(f"数据清洗完成，已保存至: {output_path}")

    except Exception as e:
        logger.exception(f"清洗过程中发生异常: {e}")