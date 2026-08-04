import os
import json
import csv

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'dataset', 'eu_ai_act_parsed.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'dataset', 'publishable')

def clean_text(text):
    """Remove any weird artifacts and ensure clean spacing."""
    if not text:
        return ""
    # Removing any stray backticks that were captured during HTML parsing
    text = text.replace('`', '')
    return text.strip()

def export_publishable_datasets():
    print(f"Reading internal dataset from {INPUT_PATH}...")
    
    if not os.path.exists(INPUT_PATH):
        print("Error: Input parsed JSON not found.")
        return
        
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    jsonl_out = os.path.join(OUTPUT_DIR, 'EU_AI_Act_RAG_Dataset.jsonl')
    csv_out = os.path.join(OUTPUT_DIR, 'EU_AI_Act_RAG_Dataset.csv')
    
    publishable_data = []
    
    for item in data:
        meta = item.get("metadata", {})
        text = clean_text(item.get("text", ""))
        chunk_id = item.get("id", "")
        
        doc_section = meta.get("type", "unknown").capitalize()
        
        # Extract section number and part
        section_number = ""
        part = ""
        if doc_section == "Article":
            section_number = meta.get("article_num", "")
            part = meta.get("paragraph", "")
        elif doc_section == "Annex":
            section_number = meta.get("annex_num", "")
            part = meta.get("part", "")
            
        word_count = len(text.split())
        
        # The official Eur-Lex URL for the AI Act
        source_url = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj"
        
        row = {
            "chunk_id": chunk_id,
            "document_section": doc_section,
            "section_number": section_number,
            "paragraph_or_part": part,
            "text": text,
            "word_count": word_count,
            "source_url": source_url
        }
        publishable_data.append(row)

    # Export to JSONL
    print(f"Exporting to {jsonl_out}...")
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for row in publishable_data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            
    # Export to CSV for maximum compatibility
    print(f"Exporting to {csv_out}...")
    with open(csv_out, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ["chunk_id", "document_section", "section_number", "paragraph_or_part", "word_count", "source_url", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in publishable_data:
            writer.writerow(row)
            
    print(f"Successfully exported {len(publishable_data)} records in both JSONL and CSV formats.")

if __name__ == "__main__":
    export_publishable_datasets()
