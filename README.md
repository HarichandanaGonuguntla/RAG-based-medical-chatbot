# RAG based medical chatbot

![](https://github.com/HarichandanaGonuguntla/RAG-based-medical-chatbot/blob/main/medibot.png)

This project is about an "Ask Medi Chatbot!", a web-based conversational AI application built using Streamlit. Its core functionality is a Retrieval-Augmented Generation (RAG) system, powered by Langchain. This system answers user queries by first retrieving relevant information from a FAISS vector store (which uses sentence-transformers/all-MiniLM-L6-v2 for embeddings), and then uses a HuggingFace Endpoint with the HuggingFaceH4/zephyr-7b-beta model to generate coherent responses. The chatbot also enhances user understanding by displaying snippets of the source documents, along with their original source and page numbers.

🧠 **Project Overview:**

Here’s how it works:

🗂 **FAISS Vector Store:** Medical documents are embedded using a sentence-transformer model and stored in a FAISS vector database. This allows the chatbot to semantically search for the most relevant documents based on user queries.

🤖 **HuggingFace Zephyr-7B Model:** The retrieved content is passed to the HuggingFaceH4/zephyr-7b-beta model hosted via Hugging Face Inference Endpoints, which then generates a coherent and human-like response.

🔄 **LangChain Orchestration:** LangChain coordinates the retrieval and generation process and ensures that the prompt structure encourages grounded, context-relevant answers.

🌐 **Streamlit Frontend:** Provides an interactive and responsive web-based UI for real-time conversations with the chatbot, maintaining session history.

🧰 **Tech Stack:**

**LangChain:** Manages prompt flow, context injection, and chaining logic for RAG.

**FAISS:** Fast, efficient vector store for semantic document retrieval.

**HuggingFace Inference API:** Hosts the HuggingFaceH4/zephyr-7b-beta model for response generation.

**Streamlit:** Delivers an intuitive web interface for seamless interaction.

**HuggingFaceEmbeddings:** Used to embed documents into vector space using the sentence-transformers/all-MiniLM-L6-v2 model.


