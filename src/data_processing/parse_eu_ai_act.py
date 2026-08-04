import os
import json
import re
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chroma_db')
HTML_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset', 'L_202401689EN.000101.fmx.xml.html')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\uFFFD', ' ')
    return text.strip()

def parse_and_store():
    print(f"Loading HTML from {HTML_PATH}...")
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        
    print("Initializing ChromaDB...")
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    LOCAL_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'embedding_model', 'all-MiniLM-L6-v2'))
    model_name = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "all-MiniLM-L6-v2"
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    
    collection = client.get_or_create_collection(
        name="eu_ai_act_context",
        embedding_function=embedding_fn,
        metadata={"description": "EU AI Act Articles and Annexes"}
    )
    
    ids = []
    documents = []
    metadatas = []
    
    print("Parsing Articles and Annexes...")
    # Find all articles and annexes
    subdivisions = soup.find_all('div', class_='eli-subdivision')
    for sub in subdivisions:
        sub_id = sub.get('id', '')
        
        # Determine if it's an article or annex
        if sub_id.startswith('art_'):
            art_num = sub_id.split('_')[1]
            title_div = sub.find('div', class_='eli-title')
            title = clean_text(title_div.text) if title_div else f"Article {art_num}"
            
            # Find sub-paragraphs like 010.001
            paragraphs = sub.find_all('div', id=re.compile(r'^\d{3}\.\d{3}$'))
            if paragraphs:
                for p in paragraphs:
                    p_id = p.get('id')
                    try:
                        p_num = int(p_id.split('.')[1])
                    except:
                        p_num = p_id
                        
                    doc_id = f"art_{art_num}_{p_num}"
                    text = clean_text(p.text)
                    if not text:
                        continue
                        
                    doc_text = f"Article {art_num}.{p_num}: {title}\n\n{text}"
                    ids.append(doc_id)
                    documents.append(doc_text)
                    metadatas.append({"type": "article", "article_num": str(art_num), "paragraph": str(p_num)})
            else:
                # If no standard paragraphs, chunk by top-level <p> or just take the whole thing
                text = clean_text(sub.text)
                if text:
                    doc_id = f"art_{art_num}_all"
                    ids.append(doc_id)
                    documents.append(f"Article {art_num}: {title}\n\n{text}")
                    metadatas.append({"type": "article", "article_num": str(art_num), "paragraph": "all"})
                    
        elif sub_id.startswith('anx_'):
            anx_num = sub_id.split('_')[1]
            title = f"Annex {anx_num}"
            
            # For annexes, try to split by sections or just top-level <p> to avoid huge chunks
            # We'll split by <p> and group them into ~300 word chunks
            paragraphs = sub.find_all('p')
            current_chunk = []
            current_words = 0
            chunk_idx = 1
            
            for p in paragraphs:
                text = clean_text(p.text)
                if not text: continue
                words = len(text.split())
                current_chunk.append(text)
                current_words += words
                
                if current_words > 300:
                    chunk_text = " ".join(current_chunk)
                    doc_id = f"anx_{anx_num}_{chunk_idx}"
                    ids.append(doc_id)
                    documents.append(f"{title} (Part {chunk_idx})\n\n{chunk_text}")
                    metadatas.append({"type": "annex", "annex_num": str(anx_num), "part": str(chunk_idx)})
                    
                    current_chunk = []
                    current_words = 0
                    chunk_idx += 1
                    
            # Add remaining
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                doc_id = f"anx_{anx_num}_{chunk_idx}"
                ids.append(doc_id)
                documents.append(f"{title} (Part {chunk_idx})\n\n{chunk_text}")
                metadatas.append({"type": "annex", "annex_num": str(anx_num), "part": str(chunk_idx)})

    print(f"Extracted {len(documents)} chunks. Ensuring unique IDs...")
    
    unique_ids = []
    seen = set()
    for doc_id in ids:
        base = doc_id
        counter = 1
        while doc_id in seen:
            doc_id = f"{base}_{counter}"
            counter += 1
        seen.add(doc_id)
        unique_ids.append(doc_id)
        
    print("Upserting to ChromaDB...")
    
    # Upsert in batches to avoid overwhelming the db or memory
    batch_size = 100
    for i in range(0, len(unique_ids), batch_size):
        end = min(i + batch_size, len(unique_ids))
        collection.upsert(
            ids=unique_ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"Upserted {end}/{len(ids)} chunks...")
        
    print("Done!")
    
    # Export to JSON
    export_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset', 'eu_ai_act_parsed.json')
    export_data = []
    for i in range(len(unique_ids)):
        export_data.append({
            "id": unique_ids[i],
            "text": documents[i],
            "metadata": metadatas[i]
        })
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)
    print(f"Exported parsed dataset to {export_path}")

if __name__ == "__main__":
    parse_and_store()
