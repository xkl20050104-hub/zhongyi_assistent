import os
from loguru import logger
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage, PromptTemplate
)
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# 配置 Loguru 日志输出格式（可选）
logger.add("logs/project.log", rotation="10 MB", encoding="utf-8")

from llama_index.core.node_parser import SentenceSplitter

#指定全局 llm 与 embedding模型
def configure_llm():
    """专门负责 LLM 和 Embedding 的连接"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    Settings.llm = DashScope(
        model_name=DashScopeGenerationModels.QWEN_MAX,
        api_key=api_key,
        temperature=0
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
        api_key=api_key
    )
    logger.info("模型资源连接成功")


def configure_parsing(chunk_size=512, chunk_overlap=50):
    """专门负责文档解析策略的配置"""
    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    logger.info(f"切片策略已就绪: Size={chunk_size}, Overlap={chunk_overlap}")


def get_query_engine(data_dir="./data", persist_dir="./data/storage"):
    """向量数据库存取逻辑"""
    configure_llm()
    configure_parsing()

    if not os.path.exists(persist_dir):
        logger.warning(f"本地索引不存在，开始构建向量库...")

        # SimpleDirectoryReader 内部其实也支持显示进度
        documents = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=[".txt"]
        ).load_data()

        logger.info(f"已加载 {len(documents)} 条文档片段，开始生成 Embedding...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)  # LlamaIndex 内置 tqdm 支持

        index.storage_context.persist(persist_dir=persist_dir)
        logger.success(f"向量库已持久化至: {persist_dir}")
    else:
        logger.info(f"发现本地索引，正在从 {persist_dir} 加载...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    # 查询引擎配置
    qa_prompt_tmpl = PromptTemplate(
        "上下文信息如下：\n{context_str}\n请根据上述内容回答：{query_str}\n回答："
    )
    query_engine = index.as_query_engine(similarity_top_k=3)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    return query_engine