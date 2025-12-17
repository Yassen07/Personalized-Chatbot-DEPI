import os
import time
from typing import List, Dict, Any, Optional
import statistics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random # For selecting a random sample

# LangChain & RAG Imports
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import CSVLoader 

# =================================================================
# 1. CONFIGURATION (Matching your API setup)
# =================================================================

# IMPORTANT: Ensure these relative paths are correct when running the script!
GGUF_FILENAME = "gemma-2b-it.Q4_K_M.gguf"
GGUF_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", GGUF_FILENAME))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db_gemma_llama_cpp"))

CSV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "QA_RAG", "train_complete_deduplicated.csv"))

# New configuration for sampling
SAMPLE_SIZE = 1000 # The maximum number of queries to process

# =================================================================
# 2. DATA LOADING & HELPER FUNCTIONS
# =================================================================

def print_resolved_paths():
    """Prints the full resolved paths for debugging file locations."""
    print("--- Resolved File Paths for Debugging ---")
    print(f"GGUF Model Path: {GGUF_MODEL_PATH}")
    print(f"Vector DB Path: {DB_PATH}")
    print(f"CSV Data Path: {CSV_FILE_PATH}")
    print("-----------------------------------------")


def load_training_data(file_path: str = CSV_FILE_PATH) -> List[Document]:
    """Loads the original training data from the CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data CSV not found at the expected path: {file_path}. Please verify 'train_complete.csv' exists.")
    
    print(f"Loading evaluation data from {file_path}...")
    
    loader = CSVLoader(
        file_path=file_path, 
        encoding="utf-8", 
        csv_args={'delimiter': ','},
        metadata_columns=['question', 'answer']
    )
    return loader.load()

def format_docs(docs: List[Document]) -> str:
    """Formats the retrieved documents into a string for the LLM context."""
    formatted_solutions = []
    for doc in docs:
        past_solution = doc.page_content
        source_question = doc.metadata.get('question', 'N/A (No Source Q)')
        expected_answer = doc.metadata.get('answer', 'N/A (No Expected A)')
        
        formatted_solutions.append(
            f"Source Question: {source_question}\n"
            f"Expected Answer (for comparison): {expected_answer}\n"
            f"Indexed Solution Content:\n{past_solution}"
        )
    return "\n---\n".join(formatted_solutions)

def calculate_semantic_similarity(s1: str, s2: str, embedder: HuggingFaceBgeEmbeddings) -> float:
    """Calculates cosine similarity between the embeddings of two strings."""
    if not s1 or not s2: 
        return 0.0
    
    # Generate embeddings for both strings
    embeddings = embedder.embed_documents([s1, s2])
    
    # Reshape the embeddings array for scikit-learn's cosine_similarity
    embeddings_array = np.array(embeddings).reshape(2, -1)
    
    # Calculate cosine similarity and return the single relevant score
    # [0, 1] gives the similarity between the first and second vector
    return cosine_similarity(embeddings_array)[0, 1]


# =================================================================
# 3. RAG PIPELINE INITIALIZATION
# =================================================================

def initialize_rag_components() -> Optional[Dict[str, Any]]:
    """Initialize LLM, Embeddings, Retriever, and RAG Chain."""
    
    # --- 3a. Embedding Model ---
    embedding_model = None
    try:
        print("\n[SETUP] Loading embedding model (BAAI/bge-small-en-v1.5)...")
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return None

    # --- 3b. Vector Store Loading ---
    retriever = None
    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Vector DB not found at {DB_PATH}. Cannot proceed.")
            
        print(f"[SETUP] Loading existing vector store from {DB_PATH}...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ Vector store and retriever loaded.")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading ChromaDB: {e}")
        return None

    # --- 3c. LLM Initialization (LlamaCpp) ---
    llm = None
    try:
        print(f"[SETUP] Loading LLM via LlamaCpp from {GGUF_MODEL_PATH}...")
        if not os.path.exists(GGUF_MODEL_PATH):
            raise FileNotFoundError(f"GGUF model file not found at: {GGUF_MODEL_PATH}")
            
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            temperature=0.5,
            max_tokens=2048,
            n_ctx=8192, # Increased context window to 8192 tokens
            n_gpu_layers=-1,
            verbose=False,
        )
        print("✅ LLM initialized via LlamaCpp.")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading LlamaCpp model: {e}")
        return None

    # --- 3d. RAG Chain Construction ---
    template = """You are a helpful Microsoft Technincal Support Agent. Your goal is to answer
the user's question accurately using only the 'Past Solutions' provided below.

Context (Past Solutions): {context}

User Question: {question}

Your Response:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Note: If LLM failed to load (llm is None), this chain construction would have
    # failed and returned None in step 3c. This only executes if llm is defined.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG Chain successfully constructed.")
    
    # Return embedding model to be used for semantic scoring
    return {"rag_chain": rag_chain, "retriever": retriever, "embedding_model": embedding_model}

# =================================================================
# 4. EVALUATION EXECUTION
# =================================================================

if __name__ == '__main__':
    print_resolved_paths() 
    
    components = initialize_rag_components()
    
    if not components:
        print("\n*** Aborting evaluation due to setup failure. ***")
        exit()

    rag_chain = components["rag_chain"]
    retriever = components["retriever"]
    # Extract the embedding model for the new semantic scoring metric
    embedding_model = components["embedding_model"] 
    all_results = []
    
    try:
        test_documents = load_training_data()
    except Exception as e:
        print(f"\n*** Aborting evaluation: Error loading training data: {e} ***")
        exit()
        
    # --- SAMPLING LOGIC ---
    initial_document_count = len(test_documents)
    
    if initial_document_count > SAMPLE_SIZE:
        print(f"\n[SAMPLING] Selecting a random sample of {SAMPLE_SIZE} queries from {initial_document_count} total documents.")
        # Randomly sample documents if the dataset is larger than the requested size
        test_documents = random.sample(test_documents, SAMPLE_SIZE)
    else:
        print(f"\n[SAMPLING] Total documents ({initial_document_count}) is less than requested sample size ({SAMPLE_SIZE}). Processing all available documents.")
    # --- END SAMPLING LOGIC ---
        
    print("\n" + "="*80)
    print(f"STARTING EVALUATION: Running {len(test_documents)} queries from training data")
    print("="*80)

    # Process all documents in the test set.
    for i, doc in enumerate(test_documents[:]): 
        
        query = ''
        expected_answer = ''
        
        # 1. Clean metadata keys
        # Clean keys just in case they have leading/trailing whitespace
        clean_metadata = {k.strip(): v for k, v in doc.metadata.items()}
        
        # 2. Extract Question (used as the query)
        query = clean_metadata.get('question') or clean_metadata.get('Question', '')
        
        # 3. Extract Expected Answer
        expected_answer = clean_metadata.get('answer') or clean_metadata.get('Answer', '')
        
        if not query or not expected_answer:
             available_keys = list(doc.metadata.keys())
             print(f"\n--- TEST CASE {i+1} --- SKIPPED. ---")
             print(f"Reason: Question or expected 'answer' not found. Available keys: {available_keys}")
             continue

        print(f"\n--- TEST CASE {i+1} ---")
        print(f"QUERY: {query}")
        print(f"EXPECTED: {expected_answer[:80]}...")
        
        start_time = time.time()
        
        try:
            # 1. RETRIEVAL STEP: Get the context documents first
            retrieved_docs = retriever.invoke(query)
            context_text = format_docs(retrieved_docs)
            
            # Metric 1: Retrieval Hit Rate (Context Relevance Proxy)
            retrieval_hit = 1 if expected_answer.lower() in context_text.lower() else 0
            
            # 2. GENERATION STEP: Get the final response
            rag_response = rag_chain.invoke(query)
            
            elapsed_time = time.time() - start_time
            
            # Metric 2: Semantic Similarity Score (Answer Quality Metric)
            semantic_score = calculate_semantic_similarity(rag_response, expected_answer, embedding_model)
            
            # Metric 3: Response Length (Conciseness Metric)
            response_token_count = len(rag_response.split())

            # Store results
            all_results.append({
                "retrieval_hit": retrieval_hit,
                "semantic_score": semantic_score, 
                "response_length": response_token_count, 
                "inference_time": elapsed_time
            })
            
            # Print results for manual inspection
            print(f"\n[SCORE] Retrieval Hit: {retrieval_hit} | Semantic Score: {semantic_score:.4f} | Length: {response_token_count} words | Time: {elapsed_time:.2f}s")
            print(f"\nRAG RESPONSE:")
            print(rag_response.strip())
            
        except Exception as e:
            # Handle token limit or other runtime errors gracefully
            print(f"❌ ERROR during RAG inference for query {i+1} (likely token limit exceeded): {e}")
            
            # Record failure metrics to track the failed run
            all_results.append({
                "retrieval_hit": 0,
                "semantic_score": 0.0,
                "response_length": 0,
                "inference_time": 0.0 # Time = 0 indicates a failed run
            })
            
            # Continue the loop to process the next query
            continue
            
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - SUMMARY REPORT")
    print("="*80)
    
    if all_results:
        # Calculate summary metrics
        total_runs = len(all_results)
        
        # Filter for successful runs (time > 0) to calculate meaningful averages
        successful_results = [r for r in all_results if r["inference_time"] > 0]
        total_successful_queries = len(successful_results)
        total_failed_queries = total_runs - total_successful_queries
        
        # Initialize averages
        avg_hit_rate = 0
        avg_semantic_score = 0
        avg_length = 0
        median_length = 0
        avg_time = 0
        median_time = 0
        
        if total_successful_queries > 0:
            # Retrieval Metrics
            avg_hit_rate = sum(r["retrieval_hit"] for r in successful_results) / total_successful_queries
            
            # Answer Quality Metrics
            avg_semantic_score = sum(r["semantic_score"] for r in successful_results) / total_successful_queries
            
            # Conciseness Metrics
            lengths = [r["response_length"] for r in successful_results]
            avg_length = sum(lengths) / total_successful_queries
            median_length = statistics.median(lengths)
    
            # Efficiency Metrics
            times = [r["inference_time"] for r in successful_results]
            avg_time = sum(times) / total_successful_queries
            median_time = statistics.median(times)


        print(f"Total Queries Attempted: {total_runs}")
        print(f"Total Successful Queries: {total_successful_queries}")
        print(f"Total Failed Queries (Token Limit/Other Error): {total_failed_queries}")
        
        print(f"--- RETRIEVAL METRICS (Context Quality - Avg over success) ---")
        print(f"Average Retrieval Hit Rate: {avg_hit_rate:.4f} (Target Answer Found in Context)")
        
        print(f"--- GENERATION METRICS (Semantic Answer Quality - Avg over success) ---")
        print(f"Average Semantic Similarity Score: {avg_semantic_score:.4f} (Response vs. Expected)")
        
        print(f"--- CONCISENESS METRICS (Response Length - Avg over success) ---")
        print(f"Average Response Length: {avg_length:.1f} words")
        print(f"Median Response Length: {median_length:.1f} words")

        print(f"--- EFFICIENCY METRICS (Performance - Avg over success) ---")
        print(f"Average Inference Time: {avg_time:.2f} seconds/query")
        print(f"Median Inference Time: {median_time:.2f} seconds/query")
    else:
        print("No valid queries were processed.")
    
    print("="*80)