import os
import json
import chromadb
from chromadb.utils import embedding_functions

# Use the lightweight MiniLM model which is ~80MB and highly efficient for local use and Dockerization
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'embedding_model', 'all-MiniLM-L6-v2'))
if os.path.exists(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL_NAME = LOCAL_MODEL_PATH
else:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chroma_db')

class VectorStore:
    def __init__(self):
        # Ensure DB path exists
        os.makedirs(DB_PATH, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Collection for prompts (Golden 68 dataset)
        self.prompts_collection = self.client.get_or_create_collection(
            name="golden68_prompts",
            embedding_function=self.embedding_fn,
            metadata={"description": "Golden 68 Evaluation Prompts"}
        )
        
        # Collection for historical evaluations
        self.evals_collection = self.client.get_or_create_collection(
            name="evaluation_history",
            embedding_function=self.embedding_fn,
            metadata={"description": "Historical Model Evaluations and Scores"}
        )
        
        # Collection for EU AI Act context
        self.eu_ai_act_collection = self.client.get_or_create_collection(
            name="eu_ai_act_context",
            embedding_function=self.embedding_fn,
            metadata={"description": "EU AI Act Articles and Annexes"}
        )

    def sync_dataset(self, dataset_path: str):
        """Sync golden68.json dataset into the vector store for semantic search."""
        if not os.path.exists(dataset_path):
            return
            
        import hashlib
        
        # Calculate MD5 hash of the dataset
        with open(dataset_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        # Check if hash matches what's stored in a meta-file
        hash_file = os.path.join(DB_PATH, "dataset_hash.txt")
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                if f.read().strip() == file_hash:
                    # Dataset hasn't changed, skip sync
                    return
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            data = {"prompts": data}
            
        if "prompts" not in data:
            return
            
        prompts = data["prompts"]
        
        ids = []
        documents = []
        metadatas = []
        
        for p in prompts:
            ids.append(p.get("id", f"prompt_{len(ids)}"))
            # The prompt text is what we want to embed for semantic searching
            documents.append(p.get("prompt", ""))
            
            # Metadata for filtering
            metadatas.append({
                "pillar": p.get("pillar", "unknown"),
                "level": p.get("level", 1)
            })
            
        # Add or update in Chroma
        # Using upsert allows safe re-syncing without duplicates
        if ids:
            self.prompts_collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        # Save the new hash
        with open(hash_file, 'w') as f:
            f.write(file_hash)

    def log_evaluation(self, eval_data: dict):
        """
        Log a single evaluated prompt into the history collection.
        eval_data should contain: session_id, prompt_id, prompt_text, 
        model_name, provider, score, rationale, response_text
        """
        self.log_evaluations([eval_data])
        
    def log_evaluations(self, eval_data_list: list):
        """
        Log multiple evaluated prompts into the history collection efficiently.
        """
        if not eval_data_list:
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for eval_data in eval_data_list:
            doc_id = f"{eval_data.get('session_id')}_{eval_data.get('prompt_id')}"
            document_text = f"Prompt: {eval_data.get('prompt_text', '')}\nResponse: {eval_data.get('response_text', '')}"
            
            metadata = {
                "session_id": eval_data.get("session_id", "unknown"),
                "prompt_id": eval_data.get("prompt_id", "unknown"),
                "model_name": eval_data.get("model_name", "unknown"),
                "provider": eval_data.get("provider", "unknown"),
                "score": float(eval_data.get("score", 0.0))
            }
            
            ids.append(doc_id)
            documents.append(document_text)
            metadatas.append(metadata)
            
        self.evals_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def get_relevant_eu_ai_act_context(self, query: str, article_refs: str = None, top_k: int = 3) -> str:
        """
        Retrieve relevant EU AI Act context based on the query and referenced articles.
        query should be the expected behavior or evaluation signal.
        article_refs is a string like "Articles 10, 14, 22"
        """
        where_clause = None
        
        # Parse article references if provided
        if article_refs:
            import re
            # Extract numbers from the string e.g. "Articles 10, 14, 22" -> ["10", "14", "22"]
            article_nums = re.findall(r'\b\d+\b', article_refs)
            if article_nums:
                if len(article_nums) == 1:
                    where_clause = {"article_num": article_nums[0]}
                else:
                    where_clause = {"$or": [{"article_num": num} for num in article_nums]}
                    
        # Semantic search
        try:
            results = self.eu_ai_act_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            if not results['documents'] or not results['documents'][0]:
                return ""
                
            return "\n\n".join(results['documents'][0])
        except Exception as e:
            print(f"Error querying EU AI Act context: {e}")
            return "EU AI Act context could not be retrieved due to a system error. Please evaluate based on general knowledge."

import streamlit as st
@st.cache_resource
def get_vector_store():
    return VectorStore()
