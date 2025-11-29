
---

# 📘 **README — MS Support Chatbot (RAG + Memory)**

## 🧠 Overview

This project is a **Microsoft Support Chatbot** powered by:

* **RAG (Retrieval-Augmented Generation)**
* **LLaMA/Gemma LLM running locally via llama-cpp-python**
* **ChromaDB** vector database
* **Persistent chat memory (SQLite-based)**
* **Clean modern frontend built in HTML/CSS/JS**

The chatbot can:

✔ Answer user questions based on your custom documents
✔ Remember previous conversation history
✔ Retrieve relevant chunks from your document database
✔ Respond like a Microsoft support agent
✔ Run fully locally — *no cloud required*

---

## 🚀 Project Structure

```
project/
│
├── RAg.py                 # FastAPI backend + RAG + memory
├── chat_memory.db         # Persistent conversation memory (auto-generated)
├── chroma_db/             # Vector database folder
├── model/                 # Local LLaMA/Gemma GGUF model file
├── data/                  # Your documents for RAG indexing
│
└── frontend/
    └── index.html         # Chat UI 
```

---

## 🧩 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:

* FastAPI
* Uvicorn
* ChromaDB
* llama-cpp-python (GPU/CPU)
* sqlite3 (built-in)
* Python 3.10+

---

## 📥 Download Resources

### 🔗 **1. Model File (.gguf — LLaMA/Gemma)**

➡️ *Paste your model download link here*

```
MODEL DOWNLOAD LINK:
https://www.kaggle.com/models/omarabdulqadir1/llm
```

Place the downloaded `.gguf` file inside:

```
/model
```

---

### 🔗 **2. Chroma Vector Database (Documents Index)**

➡️ *Paste your vector DB link here*

```
VECTOR DATABASE DOWNLOAD LINK:
https://www.kaggle.com/code/omarehab9/rag-application-demo
```

Extract and place it in:

```
/chroma_db
```

---

## 🏃‍♂️ Running the Project

### 1️⃣ Start the Backend (FastAPI)

```bash
uvicorn RAg:app --reload --port 8000
```

Backend URL:

```
http://localhost:8000
```

---

### 2️⃣ Open the Frontend

Option A — Open directly

```
frontend/index.html
```

Option B — Serve it (recommended):

```bash
cd frontend
python -m http.server 5500
```

Then visit:

```
http://localhost:5500
```

---

## 🧠 How Chat Memory Works

The project uses **SQLite-based persistent memory**:

* Each user session is tracked using a `session_id`
* Every message is stored in `chat_memory.db`
* Memory is appended and loaded at every request
* Helps the model maintain context across multiple chat turns

---

## 🏗 How RAG Works

1. User sends a message
2. Memory is loaded
3. Query is passed into the RAG chain
4. ChromaDB retrieves relevant document chunks
5. LLM uses:

   * conversation history
   * retrieved knowledge
6. Generates a contextual, helpful answer

---

## 🖥 Frontend Features

* Animated typing indicator
* Modern UI with gradients and shadow effects
* Auto-scroll
* Error handling
* Smooth animations
* Ready to deploy as a static site

---

## 🔮 Future Improvements

* Add user authentication
* Add conversation summarization memory
* Add file upload for new documents
* Add Docker support
* Add dark/light theme toggle

---

## 📄 License

This project is private unless you choose to make it open source.
You may freely modify and distribute your customized version.

---
