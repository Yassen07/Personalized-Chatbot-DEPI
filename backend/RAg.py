import os
import time
import sqlite3
from typing import List, Optional
from datetime import datetime

# FastAPI & Pydantic Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# LangChain & RAG Imports
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =================================================================
# 1. CONFIGURATION
# =================================================================

GGUF_FILENAME = "gemma-2b-it.Q4_K_M.gguf"
GGUF_MODEL_PATH = "./model/gemma-2b-it.Q4_K_M.gguf"
DB_PATH = "./chroma_db_gemma_llama_cpp"

# Persistent chat memory DB file
MEMORY_DB_FILE = "./chat_memory.db"

# How many last messages to include in prompt (tuneable)
MEMORY_TAKE_LAST = 20

# =================================================================
# 2. SQLITE PERSISTENT MEMORY UTILITIES
# =================================================================

def init_memory_db(db_file: str = MEMORY_DB_FILE):
    """Initialize SQLite DB and create table if not exists."""
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def add_message_to_memory(session_id: str, role: str, content: str, db_file: str = MEMORY_DB_FILE):
    """Insert a message row into the memory DB."""
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat())
        )
        conn.commit()
    finally:
        conn.close()

def get_history_from_memory(session_id: str, limit: int = MEMORY_TAKE_LAST, db_file: str = MEMORY_DB_FILE) -> List[dict]:
    """Fetch the last `limit` messages for the session, ordered oldest->newest."""
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit)
        )
        rows = cur.fetchall()
        # We fetched newest-first; reverse to return oldest-first.
        rows.reverse()
        return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]
    finally:
        conn.close()

def clear_memory(session_id: str, db_file: str = MEMORY_DB_FILE):
    """Delete all messages for a session."""
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()

# Initialize memory DB on import
init_memory_db()

# =================================================================
# 3. HELPER FUNCTIONS (formatting)
# =================================================================

def format_docs(docs: List[Document]) -> str:
    """Formats the retrieved documents into a string for the LLM context."""
    formatted_answers = []
    for doc in docs:
        # Assuming the metadata structure from the setup cell: {"answer": "..." }
        past_q = doc.page_content
        past_a = doc.metadata.get('answer', 'No answer found')
        formatted_answers.append(f"**Past Q**: {past_q}\n**Past A**: {past_a}")
    return "\n---\n".join(formatted_answers)

def build_history_text(session_id: str) -> str:
    """Build readable text from stored history to inject into the question prompt."""
    history = get_history_from_memory(session_id)
    if not history:
        return ""  # nothing to inject
    parts = []
    for item in history:
        # We'll only include role and content. Timestamp omitted to keep prompt short.
        parts.append(f"{item['role'].upper()}: {item['content']}")
    return "\n".join(parts)

# =================================================================
# 4. RAG PIPELINE INITIALIZATION
# =================================================================

# --- 4a. Embedding Model ---
try:
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    print(f"Error loading embeddings: {e}")
    embedding_model = None

# --- 4b. Vector Store Loading ---
try:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector DB not found at {DB_PATH}. Please run the indexing steps first.")
    print(f"Loading existing vector store from {DB_PATH}...")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Vector store and retriever loaded.")
except FileNotFoundError as e:
    print(e)
    retriever = None

# --- 4c. LLM Initialization (LlamaCpp) ---
try:
    print(f"Loading LLM via LlamaCpp from {GGUF_MODEL_PATH}...")
    llm = LlamaCpp(
        model_path=GGUF_MODEL_PATH,
        temperature=0.5,
        max_tokens=2048,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("✅ LLM initialized via LlamaCpp.")
except Exception as e:
    print(f"Error loading LlamaCpp model: {e}")
    llm = None

# --- 4d. RAG Chain Construction ---
template = """You are a helpful Microsoft Technincal Support Agent. Your goal is to answer
the user's question accurately using only the 'Past Solutions' provided below.

Context (Past Solutions): {context}

User Question: {question}

Your Response:"""

prompt = ChatPromptTemplate.from_template(template)

if llm and retriever:
    # Note: we will pass a string to rag_chain.invoke(...) where the string is the 'question'.
    # The chain's "context" value comes from the retriever | format_docs expression.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG Chain successfully constructed.")
else:
    rag_chain = None
    print("❌ RAG Chain failed to construct due to missing LLM or Retriever.")

# =================================================================
# 5. FASTAPI APPLICATION
# =================================================================

app = FastAPI(
    title="LlamaCpp RAG API (with persistent memory)",
    description="Gemma-2B RAG pipeline using LlamaCpp + SQLite persistent chat memory."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class MemoryClearRequest(BaseModel):
    session_id: str

@app.get("/")
async def hello():
    return {"message": "hello on MS support (persistent memory enabled)"}

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "llm_loaded": llm is not None,
        "rag_ready": rag_chain is not None,
        "model": GGUF_FILENAME,
        "memory_db": os.path.exists(MEMORY_DB_FILE)
    }

@app.post("/chat")
def chat(req: ChatRequest):
    """Endpoint for processing user queries through the RAG chain with persistent memory injection."""
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="RAG service is unavailable. Check model and vector database loading logs."
        )

    try:
        session_id = req.session_id or "default"
        query_start = time.time()

        # 1) Build memory text from stored messages
        history_text = build_history_text(session_id)
        if history_text:
            history_block = f"Conversation History:\n{history_text}\n\n"
        else:
            history_block = ""

        # 2) Build the question string that will be passed into the RAG chain as 'question'
        # We inject the history BEFORE user message so the model sees context.
        question_for_rag = f"{history_block}{req.message}"

        # 3) Invoke the rag chain (the chain will call the retriever for 'context' automatically)
        # We pass a single string; the RunnablePassthrough maps it to the 'question' input slot.
        response = rag_chain.invoke(question_for_rag)

        query_time = time.time() - query_start

        # 4) Persist the new messages (user and assistant) into SQLite memory
        try:
            add_message_to_memory(session_id, "user", req.message)
            # Save assistant reply as well
            add_message_to_memory(session_id, "assistant", response)
        except Exception as mem_e:
            # Don't break the response if DB write fails; log and continue.
            print(f"Warning: failed to write memory for session {session_id}: {mem_e}")

        return {
            "session_id": session_id,
            "query": req.message,
            "response": response,
            "history_length": len(get_history_from_memory(session_id)),
            "inference_time_seconds": round(query_time, 2)
        }

    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model inference: {e}"
        )

@app.get("/memory/history/{session_id}")
def get_memory_history(session_id: str, limit: int = MEMORY_TAKE_LAST):
    """Return last `limit` messages for a session."""
    try:
        history = get_history_from_memory(session_id, limit=limit)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/clear")
def clear_memory_endpoint(req: MemoryClearRequest):
    """Clear memory for a session (DELETE)."""
    try:
        clear_memory(req.session_id)
        return {"session_id": req.session_id, "cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================================================================
# 6. RUN (for local testing)
# =================================================================

if __name__ == '__main__':
    print("\n\n" + "="*50)
    print("Starting FastAPI RAG Server with persistent memory...")
    print(f"LLM Ready: {llm is not None} | RAG Chain Ready: {rag_chain is not None}")
    print(f"Memory DB: {MEMORY_DB_FILE} (exists: {os.path.exists(MEMORY_DB_FILE)})")
    print("="*50)
    # Use a single worker to avoid LlamaCpp + multiprocessing issues
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
