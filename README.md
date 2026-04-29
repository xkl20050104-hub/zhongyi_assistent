# 中医诊疗助手

一个基于 RAG（检索增强生成）的中医问答项目。  
项目会先清洗中医术语文档，再构建向量索引，通过大模型结合检索结果回答问题。

## 功能特性

- 支持从 `.docx` 中提取并清洗中医语料
- 基于 `llama-index` + DashScope Embedding 构建本地向量索引
- 基于 DashScope 大模型进行问答生成
- 提供命令行交互问答（输入 `q` 退出）
- 提供 `LangSmith` 评估脚本（语义相似度自动打分）

## 项目结构

```text
.
├─ main.py                     # 主程序入口（交互式问答）
├─ experiments.py              # LangSmith 评估入口
├─ requirement.txt             # Python 依赖
├─ data/
│  ├─ 《中医临床诊疗术语第2部分：证候》（修订版）.docx
│  ├─ cleaned_corpus.txt       # 清洗后的语料（首次运行自动生成）
│  └─ storage/                 # 持久化向量索引目录（首次运行自动生成）
└─ modules/
   ├─ data_clean.py            # 语料清洗
   ├─ query_engine.py          # LLM/Embedding 配置与索引加载
   └─ work_flow.py             # RAG 工作流（检索 + 组装提示词 + 生成）
```

## 环境要求

- Python 3.9+
- Windows / macOS / Linux（当前项目在 Windows 下可运行）

## 安装步骤

1. 创建并激活虚拟环境（推荐）：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. 安装依赖：

```bash
pip install -r requirement.txt
```

3. 在项目根目录创建 `.env` 文件，至少包含：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

如果要运行 `experiments.py` 做 LangSmith 评估，请额外配置：

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=TCM_RAG
```

## 运行项目

在项目根目录执行：

```bash
python main.py
```

首次运行时会自动执行以下步骤：

1. 检查 `data/cleaned_corpus.txt` 是否存在，不存在则清洗 `.docx` 语料
2. 检查 `data/storage` 是否存在，不存在则构建并持久化向量索引
3. 进入命令行问答模式

问答示例：

```text
问：气虚证常见表现是什么？
答：...
```

输入 `q` 可退出程序。

## 运行评估实验（可选）

```bash
python experiments.py
```

说明：

- 该脚本会调用 LangSmith 的 `evaluate` 流程
- 数据集名默认为 `中医语料`（需在 LangSmith 侧提前准备）
- 评估器会让模型给出语义相似度分数，并转换为 0~1 范围

## 常见问题

- `源文件不存在`：请确认 `data/《中医临床诊疗术语第2部分：证候》（修订版）.docx` 路径正确
- `API key` 报错：检查 `.env` 中 `DASHSCOPE_API_KEY` 是否有效
- 首次启动慢：属于正常现象，正在做语料清洗和向量索引构建
- 想重建索引：删除 `data/storage` 后重新运行 `python main.py`

## 依赖清单

核心依赖如下（详见 `requirement.txt`）：

- `llama-index`
- `llama-index-llms-dashscope`
- `llama-index-embeddings-dashscope`
- `python-docx`
- `python-dotenv`
- `loguru`
- `tqdm`
