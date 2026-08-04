<div align="center">
  
# 🏆 Golden 68 AI Audit Framework 🇪🇺
### *Paving the way for LLMs in High-Stakes Applications*

![Golden 68 AI Audit Framework](https://img.shields.io/badge/Status-Active-success?style=for-the-badge) 
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge) 
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![EU AI Act](https://img.shields.io/badge/Compliance-EU_AI_Act-FFD700?style=for-the-badge)

</div>

<br>

Welcome to the **Golden 68 AI Audit Framework**! 

As Large Language Models (LLMs) become the "brains" behind critical, high-stakes applications, ensuring their safety, reliability, and legality is paramount. 

Our project is a comprehensive auditing tool designed to rigorously filter and evaluate AI applications and base LLMs to determine if they are truly "up to the mark." **Our ultimate aim is to establish this framework as a premier tool for evaluating LLMs according to the EU AI Act before its final enforcement date.**

*(Note: We are currently focusing exclusively on **text-to-text** applications, as this remains the most widely adopted use case.)*

---

## 🎯 The Core Vision: How It Works

Evaluating an LLM requires more than just reading its outputs. We have built a robust, multi-layered pipeline to rigorously test, manipulate, and audit these models:

### 1. 📝 The "Golden 68" Adversarial Dataset
We have curated a highly effective dataset of prompts—crafted meticulously by both top-tier AI and human hands. These prompts are designed to inject edge cases, attempt to manipulate the model, and aggressively test its explanations. Every prompt is paired with a **"ground truth" expected general answer** to establish a baseline for correct behavior.

### 2. ⚖️ EU AI Act Integration
We don't just test for logic; we test for legality. We have created and attached a dedicated dataset of indexed **EU AI Act Laws**. This ensures the model isn't just answering correctly, but answering *compliantly*.

### 3. 🤖 The "LLM-as-a-Judge"
We utilize the smartest, state-of-the-art LLMs available (like Gemini 1.5 Pro or Claude 3.5 Sonnet) to act as the "Judge". The Judge is fed:
*   The test model's response.
*   The ground truth / expected behavior.
*   The specific EU AI Act laws it must follow.

The Judge then critically evaluates the response, providing a strict rating and a detailed explanation of whether it adhered to the laws and the expected logic.

### 4. 🧑‍🔬 Human Researcher Reliability
To ensure our AI Judge isn't hallucinating, we incorporate human reliability. Human researchers perform the exact same grading task on the test model's answers. By comparing the Human Rating with the Judge's Rating (using statistical methods like Cohen's Kappa), we guarantee the evaluation is incredibly reliable.

### 5. 🔄 Continuous Improvement Loop (The Output)
Testing is useless without actionable feedback. At the end of the audit, both the AI Judge and the Human Auditor provide their ratings and explanations. 
Finally, our tool automatically generates a rich **Failure Dataset (JSONL)**. This dataset pinpoints exactly *where* the model went wrong and detailed explanations of *why* it went wrong. AI companies can use this exact dataset to continuously improve and fine-tune their LLMs!

---

## 🚀 Getting Started

If you want to run this audit framework locally and test your own models:

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
4. Configure your API keys in the interface, and begin your EU AI Act compliance audit!

---

## 🤝 Contributing & The Future
We are racing toward the final declaration date of the EU AI Act. We welcome researchers, developers, and policymakers to contribute to our Golden 68 dataset and help us make LLMs safer for high-stakes deployment. 
