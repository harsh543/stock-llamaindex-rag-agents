# stock-llamaindex-rag-agents

# 🤖 AI Agents Cookbooks - LlamaIndex RAG Agents

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11.6-green.svg)](https://llamaindex.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ytang07/ai_agents_cookbooks/blob/main/llamaindex/llama31_8b_rag_agent.ipynb)

> **A comprehensive collection of Retrieval Augmented Generation (RAG) agents** built with LlamaIndex, showcasing production-ready implementations across different LLM providers, vector stores, and evaluation frameworks.

## 🌟 Features

- 🧠 **Multiple LLM Providers**: OpenAI GPT-4, Llama 3.1 70B via OctoAI
- 🗄️ **Vector Store Options**: In-memory, Milvus (Docker & Lite), persistent storage
- 📊 **Evaluation & Monitoring**: Phoenix AI for real-time tracing and evaluation
- 🛠️ **Agent Types**: ReAct agents with function calling capabilities
- 📈 **Real-world Use Cases**: Financial document analysis and Q&A systems
- 🚀 **Production Ready**: Complete examples with error handling and best practices

## 🎯 Quick Demo

```python
# Build a RAG agent in just a few lines!
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Load documents and create index
docs = SimpleDirectoryReader("./data/10k/").load_data()
index = VectorStoreIndex.from_documents(docs)

# Create query engine and agent
query_engine = index.as_query_engine()
agent = ReActAgent.from_tools([query_engine], llm=OpenAI())

# Ask questions!
response = agent.chat("Who had more profit in 2021, Lyft or Uber?")
```


## ⚡ Quick Start

### 1️⃣ **Clone & Setup**

```bash
git clone https://github.com/your-username/ai_agents_cookbooks.git
cd ai_agents_cookbooks
```

### 2️⃣ **Install Dependencies**

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended - faster installs)
pip install uv && uv pip install --system -r requirements.txt
```

### 3️⃣ **Configure Environment**

```bash
cp .env.example .env
# Edit .env and add your API keys
```

```env
OPENAI_API_KEY=sk-your-openai-key-here
OCTOAI_API_KEY=your-octoai-key-here
```

### 4️⃣ **Run Your First Agent**

```bash
jupyter notebook notebooks/rag_agent.ipynb
```



## 🔧 Configuration Examples

### 🤖 LLM Providers

<details>
<summary><b>🟢 OpenAI (GPT-4)</b> - Recommended for production</summary>

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-4",
    temperature=0.1,
    max_tokens=512
)
```
**Pros**: High quality, reliable, function calling  
**Cons**: Higher cost, rate limits

</details>

<details>
<summary><b>🦙 Llama 3.1 70B (OctoAI)</b> - Great for cost optimization</summary>

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="meta-llama-3.1-70b-instruct",
    api_base="https://text.octoai.run/v1",
    api_key=os.environ["OCTOAI_API_KEY"],
    context_window=40000,
    is_function_calling_model=True
)
```
**Pros**: Lower cost, large context, open source  
**Cons**: Slightly lower quality than GPT-4

</details>

### 🗄️ Vector Stores

<details>
<summary><b>💾 In-Memory (Default)</b> - Perfect for development</summary>

```python
from llama_index.core import VectorStoreIndex

# Simple setup - no configuration needed!
index = VectorStoreIndex.from_documents(documents)
```
**Pros**: No setup required, fast for small datasets  
**Cons**: Not persistent, limited by RAM

</details>

<details>
<summary><b>🐳 Milvus (Production)</b> - Best for scale</summary>

```python
from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    host="localhost",
    port=19530,
    dim=1536,
    collection_name="documents",
    overwrite=False  # Preserve existing data
)
```
**Pros**: Highly scalable, persistent, production-ready  
**Cons**: Requires Docker setup

</details>

## 📊 Evaluation & Monitoring with Phoenix AI

### 🔍 Real-time Monitoring

All evaluation-enabled notebooks include **Phoenix AI** for comprehensive monitoring:

```python
import phoenix as px
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# Launch Phoenix dashboard
session = px.launch_app()  # Visit http://localhost:6006

# Auto-instrument your RAG pipeline
LlamaIndexInstrumentor().instrument()
```

### 📈 What I learnt:

- **🔄 Real-time Traces**: See every step of your RAG pipeline
- **📊 Performance Metrics**: Latency, token usage, error rates
- **🎯 Quality Evaluation**: Relevance, hallucination detection
- **🐛 Debug Support**: Identify bottlenecks and issues

![Phoenix Dashboard](https://docs.arize.com/phoenix/_images/phoenix-rag-traces.png)

> **💡 Pro Tip**: Always use Phoenix in production to monitor RAG quality and catch issues early!

## 🐳 Docker Setup (Milvus)

### Quick Start with Docker

```bash
# Start Milvus vector database
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs if needed
docker-compose logs milvus

# Stop when done
docker-compose down
```

## 🔧 Troubleshooting

### Common Issues

**🔑 Missing API Keys**
```bash
Error: The OPENAI_API_KEY environment variable is not set
```
**Solution**: Ensure your `.env` file is properly configured

**🐳 Milvus Connection Issues**
```bash
MilvusException: <MilvusException: (code=2, message=Fail connecting to server)>
```
**Solution**: Ensure Docker services are running with `docker-compose up -d`

**💾 Memory Issues with Large Documents**
```bash
OutOfMemoryError
```
**Solution**: Reduce chunk size or use a more efficient vector store

### Debug Mode

Enable verbose logging for debugging:
```python
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,  # Enable debug output
    max_turns=10
)
```

## 🚀 Getting Started Examples

### Basic Financial Q&A
```python
response = agent.chat("What was Lyft's revenue growth in 2021?")
# Returns: "Lyft's revenue growth in 2021 was 36%"
```

### Multi-step Analysis
```python
response = agent.chat("Compare the profitability of Lyft and Uber in 2021")
# Agent will:
# 1. Query Lyft's net income
# 2. Query Uber's net income  
# 3. Compare and provide analysis
```

### Mathematical Operations
```python
response = agent.chat("Calculate (14 + 12)² step by step")
# Agent will:
# 1. Add 14 + 12 = 26
# 2. Square the result = 676
```





## 🏆 Acknowledgments

- **LlamaIndex Team** for the amazing RAG framework
- **Arize Phoenix** for evaluation and monitoring tools
- **AI Agents Cookbook** practice repo for understanding RAG agents
- **OctoAI** for accessible LLM inference
- **Milvus** for scalable vector storage
- **Community Contributors** for feedback and improvements

---

**⭐ If this repository helped you, please give it a star!**

> **Note**: This is a collection of educational examples and production patterns. For enterprise deployments, additional considerations around security, scalability, and compliance should be implemented.
