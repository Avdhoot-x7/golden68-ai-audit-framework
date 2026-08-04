# 🏆 Golden 68 AI Audit Framework

![Golden 68 AI Audit Framework](https://img.shields.io/badge/Status-Active-success) ![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-purple)

The **Golden 68 AI Audit Framework** is a cutting-edge, explainable AI (XAI) evaluation platform designed to rigorously audit Large Language Models (LLMs) against predefined safety, causality, and regulatory compliance standards. 

Built with an **LLM-as-a-Judge** architecture and augmented by human-in-the-loop verification, it provides an enterprise-grade pipeline for ensuring models adhere to strict operational guidelines, including alignment with the **EU AI Act**.

---

## ✨ Key Features

*   ⚖️ **LLM-as-a-Judge Pipeline:** Automate the evaluation of model responses using a highly structured judge prompt that grades based on Accuracy, Completeness, Reasoning Quality, and Compliance.
*   🧠 **Multi-Provider Support:** Seamlessly connect to leading API providers, including **NVIDIA NGC**, **OpenAI**, **Anthropic**, and **OpenRouter**, utilizing a robust adapter pattern.
*   🇪🇺 **EU AI Act Ready:** Architecture prepared for Retrieval-Augmented Generation (RAG) to dynamically inject regulatory context directly into the evaluation rubric.
*   📊 **Statistical Reliability (Cohen's Kappa):** Validate the LLM Judge's determinations against human expert audits using rigorous statistical agreement metrics (Cohen's Kappa) to ensure scoring consistency.
*   🗄️ **Persistent Vector Storage:** Powered by **ChromaDB**, securely storing historical evaluations, maintaining dataset hashes, and enabling semantic search across evaluation history.
*   📑 **Advanced Reporting:** Automatically generate comprehensive **PDF Audit Reports** detailing model vulnerabilities, compliance heatmaps, and aggregate scores.
*   🛠️ **Fine-Tuning Export:** Extract failed interactions into perfectly formatted **JSONL datasets** for continuous model alignment and fine-tuning.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10 or higher
*   API Keys for the models you wish to test (e.g., NVIDIA, OpenAI)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Avdhoot-x7/golden68-ai-audit-framework.git
    cd golden68-ai-audit-framework
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 🖥️ System Architecture

1.  **Dataset Ingestion:** Securely loads the `golden68.json` dataset, applying safe-parsing to handle incomplete or malformed entries gracefully.
2.  **Model Execution:** The target model (e.g., Llama-3, DeepSeek) generates responses to the dataset prompts via secure `ModelAdapters`.
3.  **Judge Evaluation:** The Judge LLM (e.g., Gemini, Claude) evaluates the target model's response against the expected behavior.
4.  **Human Audit:** Human experts review the Judge's scores via the Streamlit UI, providing ground-truth validation.
5.  **Statistical Validation:** The framework calculates **Cohen's Kappa** to determine the reliability between the AI Judge and the Human Auditor.
6.  **Reporting & Persistence:** Results are logged to ChromaDB, and actionable PDF/JSONL reports are generated.

---

## 📁 Project Structure

```text
golden68_framework/
├── app.py                      # Main Streamlit Application Entry Point
├── requirements.txt            # Python Dependencies
├── data/                       # Local Storage
│   ├── dataset/                # Golden68 dataset files (JSON/CSV)
│   ├── chroma_db/              # Persistent Vector Store
│   ├── reports/                # Generated PDF and JSON reports
│   └── fine_tuning/            # Exported JSONL datasets for alignment
└── src/                        # Core Application Logic
    ├── database/               # ChromaDB Vector Store Implementation
    ├── evaluation/             # Core Scoring Logic & Dataset Loaders
    ├── judges/                 # LLM-as-a-Judge Prompting and Logic
    ├── models/                 # API Adapters (NVIDIA, Anthropic, etc.)
    ├── reporting/              # PDF Generation and JSONL Exporters
    └── validation/             # Statistical metrics (Cohen's Kappa)
```

---

## 📝 Configuration

Configure your API keys directly within the Streamlit UI under the **"API Configuration"** tab. The framework utilizes aggressive error handling to manage rate limits, API exhaustion, and safety filter blocks natively.

---

## 🤝 Contributing

Contributions are welcome! If you are interested in improving the evaluation rubrics, adding new API adapters, or enhancing the statistical validation metrics, please open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
