# EU AI Act - RAG & Fine-Tuning Dataset

## Dataset Description

- **Repository / Creator:** An Independent AI Safety Research Initiative
- **Language:** English
- **License:** Open Access (Derived from Official EU Legal Texts)
- **Format:** JSON Lines (`.jsonl`) & CSV (`.csv`)

### Dataset Summary
This dataset contains the text of the **European Union Artificial Intelligence Act (EU AI Act)** meticulously parsed, cleaned, and chunked. It is specifically designed to be immediately usable by AI researchers, developers, and compliance engineers building **Retrieval-Augmented Generation (RAG) systems** or fine-tuning models for legal tech and AI compliance.

The official, complex HTML of the EU AI Act has been broken down into complete, semantically coherent chunks (individual paragraphs for Articles, and ~300-word logical chunks for Annexes), eliminating the need for developers to build custom web-scrapers or text-chunking pipelines.

### Supported Tasks and Leaderboards
- **Retrieval-Augmented Generation (RAG):** Retrieve specific legal articles to answer compliance questions.
- **Text Classification / NER:** Tagging AI risk levels based on EU AI Act definitions.
- **Legal Fine-Tuning:** Teaching LLMs the structure and rules of the EU AI Act.

## Dataset Structure

The dataset is provided in both **JSONL** and **CSV** formats to ensure maximum compatibility across all ML frameworks (Hugging Face Datasets, Pandas, LangChain, LlamaIndex, etc.).

### Data Fields

Every record (row) in the dataset contains the following fields:

- **`chunk_id`** *(string)*: A unique identifier for the chunk (e.g., `art_10_2` or `anx_3_1`).
- **`document_section`** *(string)*: Indicates whether the chunk belongs to an `Article` or an `Annex`.
- **`section_number`** *(string)*: The specific Article number or Annex number.
- **`paragraph_or_part`** *(string)*: The specific paragraph (for articles) or part number (for annexes).
- **`word_count`** *(integer)*: The number of words in this specific chunk (useful for managing LLM token limits).
- **`source_url`** *(string)*: A direct link to the official Eur-Lex publication of the EU AI Act.
- **`text`** *(string)*: The clean, parsed text of the regulation chunk.

### Example Record (JSONL)

```json
{
  "chunk_id": "art_10_1",
  "document_section": "Article",
  "section_number": "10",
  "paragraph_or_part": "1",
  "word_count": 85,
  "source_url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
  "text": "Article 10.1: Data and data governance\n\n1. High-risk AI systems which make use of techniques involving the training of models with data shall be developed on the basis of training, validation and testing data sets that meet the quality criteria referred to in paragraphs 2 to 5."
}
```

## Attribution and Citation
This dataset was extracted, parsed, and structured by an **Independent AI Safety Research Initiative** focused on rigorous AI auditing and compliance evaluation. 

If you use this dataset in your research or product, we encourage you to mention that the data was structured by this independent initiative.

## Source Data
The raw text was sourced from the official publication of the EU AI Act on [Eur-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). All text belongs to the European Union and is subject to standard EU legal notices.
