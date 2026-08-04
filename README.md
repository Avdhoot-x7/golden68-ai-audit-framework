# 🏆 Golden 68 AI Audit Framework

![Golden 68 AI Audit Framework](https://img.shields.io/badge/Status-Active-success) ![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-purple)

Welcome to the **Golden 68 AI Audit Framework**! 

This project is an easy-to-use but powerful tool designed to test and audit Large Language Models (LLMs). As AI models get smarter, we need to make sure they are safe, reliable, and follow legal rules. This framework acts as a strict "examiner" to test how well AI models behave when asked tricky, sensitive, or complex questions.

---

## 🌟 What is this project?
When companies build AI chatbots, they need to know if the bot will accidentally give dangerous advice, break laws, or make logical mistakes. 

Our framework solves this problem by using an **LLM-as-a-Judge** approach. Instead of humans reading thousands of AI responses to check for safety, we use a highly intelligent "Judge AI" (like Gemini or Claude) to automatically grade the responses of the "Test AI" (like Llama-3 or DeepSeek). We then use a statistical math formula called **Cohen's Kappa** to prove that our Judge AI grades just as accurately as a human expert would.

### How this helps others:
* **For Researchers:** It provides a scientifically proven, statistical way to measure AI safety.
* **For Companies:** It helps them quickly test their AI models against legal frameworks (like the EU AI Act) before releasing them to the public.
* **For Developers:** It automatically saves any failed responses into a special file format (`JSONL`) so developers can easily retrain and fix their models!

---

## 📂 Our Datasets

This framework relies on two very important datasets to test the AI models:

### 1. The "Golden 68" Prompt Dataset
This is our core testing dataset. It contains 68 highly specific, carefully crafted test prompts. These prompts are divided into specific "pillars" (categories) and "levels" of difficulty.
* **Safety Prompts:** Testing if the AI refuses to give harmful instructions (e.g., "How do I build a weapon?").
* **Causality Prompts:** Testing if the AI understands cause and effect logically.
* **Consistency Prompts:** Testing if the AI gives the same logical answer when asked the same question in a different way.
* Each prompt in the dataset includes the **Expected Behavior** (exactly what the AI *should* do), so the Judge knows how to grade it.

### 2. The EU AI Act (RAG Indexed Dataset)
To ensure the AI models comply with real-world laws, we have integrated the **European Union AI Act (2026)** into the system. 
Instead of forcing the Judge AI to read the entire massive law book every time it grades a response, we use a smart database (ChromaDB). This database instantly searches for only the specific legal articles relevant to the current question, and feeds just that small snippet to the Judge. This makes the grading highly legally accurate and saves a massive amount of computing power!

---

## ⚙️ How the Testing Flow Works

Here is the step-by-step journey of how an AI is tested in our framework:

1. **Load the Prompt:** The system picks a tricky question from the Golden 68 dataset.
2. **Test the Model:** The system sends the question to the AI we are testing (using API connections like NVIDIA, OpenAI, or Anthropic).
3. **The AI Answers:** The tested AI generates its response.
4. **Retrieve Laws (RAG):** The system searches the ChromaDB database for any relevant EU AI Act laws related to the topic.
5. **The Judge Evaluates:** The framework sends the original question, the expected safe behavior, the relevant EU laws, and the AI's answer to the "Judge LLM".
6. **Scoring:** The Judge gives a score from 1 to 10, determines a PASS/FAIL, and writes a detailed explanation of *why* it gave that score.
7. **Human Verification:** A human can review the Judge's score in the user interface. The system calculates the **Cohen's Kappa** score to prove the AI Judge is reliable.

---

## 📑 Results and Reporting

Once the test is finished, the framework automatically generates detailed results:

* **PDF Audit Reports:** A beautiful, easy-to-read PDF document is created. It shows a visual "heatmap" of how well the AI performed across different safety pillars. It acts as an official "Safety Certificate" or "Audit Report" for the model.
* **JSONL Fine-Tuning Export:** If the tested AI fails any questions, the system grabs those failures and perfectly formats them into a `.jsonl` file. Developers can plug this file directly into fine-tuning software to retrain the AI and fix its bad behavior!

---

## 🚀 Getting Started

If you want to run this audit framework on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Avdhoot-x7/golden68-ai-audit-framework.git
   cd golden68-ai-audit-framework
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```bash
   streamlit run app.py
   ```
4. Put in your API keys in the interface, and start auditing!

---

## 🤝 Contributing
We welcome contributions! If you have ideas for new tricky prompts for the Golden 68 dataset, or ways to improve the Judge's accuracy, please feel free to open a Pull Request.
