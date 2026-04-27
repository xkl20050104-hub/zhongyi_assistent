import os
from langsmith import traceable
from loguru import logger
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage
)
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.node_parser import SentenceSplitter


def configure_settings():
    """全局配置 LLM 和模型参数"""
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
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


@traceable(run_type="chain", name="Load_Vector_Index")
def get_index(data_dir="./data", persist_dir="./data/storage"):
    """构建或加载向量索引"""
    configure_settings()

    if not os.path.exists(persist_dir):
        logger.info("构建新索引中...")
        documents = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".txt"]).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        logger.debug("加载本地持久化索引...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index