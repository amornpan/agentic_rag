# Agentic RAG

**Agentic RAG** (by amornpan) is a robust framework for building **Retrieval-Augmented Generation (RAG)** systems with **agentic capabilities**. It enhances intelligent retrieval, fine-tuned LLM orchestration, and seamless enterprise integration.

## 🚀 Features
- **Agentic Capabilities** – Automates retrieval and response generation intelligently.
- **Adaptive Retrieval** – Dynamically fetches relevant knowledge for precise responses.
- **Fine-Tuned LLM Orchestration** – Enhances model efficiency and accuracy.
- **Enterprise Integration** – Connects structured and unstructured data sources.

## 📖 Documentation
Refer to the [official repository](https://github.com/amornpan/agentic-rag) for setup instructions, API details, and implementation guidelines.

## 🛠️ Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/amornpan/agentic-rag.git
cd agentic-rag
npm install

const { AgenticRetriever } = require('agentic-rag');

const retriever = new AgenticRetriever();

async function run() {
    const response = await retriever.query("What is the impact of climate change?");
    console.log(response);
}

run();

export OPENAI_API_KEY="your-api-key"
export VECTOR_DB_URL="your-vector-database-url"


