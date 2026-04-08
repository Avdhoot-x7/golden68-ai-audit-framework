# Golden 68 - AI Compliance & Audit Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive evaluation framework for Large Language Models (LLMs) focusing on three critical pillars: **Causality**, **Compliance**, and **Consistency**. This framework provides both automated LLM-as-Judge evaluation and human expert verification capabilities to assess AI model performance against regulatory standards, particularly the EU AI Act 2026.

## Table of Contents

- [Overview](#overview)
- [The Golden 68 Dataset](#the-golden-68-dataset)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
- [API Configuration](#api-configuration)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Evaluation Metrics](#evaluation-metrics)
- [Results & Reports](#results--reports)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Golden 68 Framework is designed to provide rigorous, multi-dimensional evaluation of Large Language Models through 68 carefully crafted prompts. It addresses critical concerns in AI deployment including:

- **Logical Reasoning**: Testing causal inference and if-then relationships
- **Regulatory Compliance**: Mapping to EU AI Act 2026 requirements
- **Response Consistency**: Measuring stability across prompt variations

This framework is ideal for:
- AI researchers evaluating model capabilities
- Organizations assessing LLMs for production deployment
- Compliance teams validating regulatory adherence
- Academic institutions studying AI safety and reliability

## The Golden 68 Dataset

### Three Core Pillars

| Pillar | Description | Prompt Count | Focus Area |
|--------|-------------|--------------|------------|
| **Causality** | Logical If-Then relationships and causal reasoning | 23 prompts | Tests model's ability to understand cause-effect relationships and logical implications |
| **Compliance** | EU AI Act 2026 regulatory mapping | 23 prompts | Evaluates adherence to transparency, safety, and documentation requirements |
| **Consistency** | Response stability across rephrasing | 22 prompts | Measures output consistency when prompts are semantically equivalent but differently worded |

### Complexity Levels

Each prompt is classified into one of five complexity levels:

- **Level 1**: Basic logic check - Simple yes/no reasoning
- **Level 2**: Simple reasoning - Straightforward inference tasks
- **Level 3**: Multi-step reasoning - Requires connecting multiple logical steps
- **Level 4**: Complex adversarial - Challenging scenarios with edge cases
- **Level 5**: Multi-step adversarial stress test - Maximum complexity with adversarial framing

## Key Features

### 1. Dual Evaluation System

- **LLM-as-Judge**: Automated evaluation using advanced LLMs (Gemini, GPT-4, Claude) as judges
- **Human Audit Lab**: Interactive interface for expert verification and validation
- **Agreement Delta Analysis**: Statistical comparison between automated and human assessments

### 2. Comprehensive Reporting

- Detailed evaluation logs with reasoning for each prompt
- Pillar-by-pillar performance breakdown
- Complexity level analysis
- Regulatory readiness heatmaps
- Visual score distributions and trend charts

### 3. Cost Tracking & Smart Resume

- Real-time API usage monitoring
- Cost estimation per provider
- Rate limit tracking
- Checkpoint system for resuming interrupted evaluations
- Multi-API key fallback support

### 4. Model Comparison & Leaderboard

- Compare multiple models side-by-side
- Historical evaluation tracking
- Weighted scoring system
- Export results in multiple formats

### 5. Interactive Web Dashboard

- Built with Streamlit for easy deployment
- No command-line expertise required
- Real-time progress tracking
- Professional visualizations with Plotly

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for the LLM providers you wish to use

### Step-by-Step Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/golden68_framework.git
cd golden68_framework
```

2. **Create a virtual environment** (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify installation**

```bash
python -c "import streamlit; import plotly; print('Installation successful!')"
```

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a rapid getting-started guide.

## How to Run

### Starting the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run app.py --server.port 8080
```

## API Configuration

The framework supports multiple LLM providers. You'll need API keys for the providers you wish to use.

### Supported Providers

| Provider | Models Supported | API Key Required | Rate Limits |
|----------|------------------|------------------|-------------|
| **Google Gemini** | gemini-2.0-flash, gemini-1.5-pro, gemini-2.0-pro | Yes - [Get Key](https://makersuite.google.com/app/apikey) | 15 RPM, 1M TPM |
| **OpenAI** | gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo | Yes - [Get Key](https://platform.openai.com/api-keys) | 500 RPM, 150K TPM |
| **Anthropic** | claude-3-opus, claude-3-sonnet, claude-3-haiku | Yes - [Get Key](https://console.anthropic.com/) | 50 RPM, 100K TPM |
| **OpenRouter** | Multiple models via unified API | Yes - [Get Key](https://openrouter.ai/keys) | 200 RPM, 200K TPM |
| **NVIDIA** | openai/gpt-oss-120b, llama-3.1-405b | Yes - [Get Key](https://build.nvidia.com/) | 120 RPM, 500K TPM |

### Configuration Steps

1. Obtain API keys from your chosen provider(s)
2. In the web interface, navigate to the **API Configuration** section
3. Enter your API keys for:
   - **Judge LLM**: The model that will evaluate responses
   - **Test LLM**: The model being evaluated
4. (Optional) Add backup API keys for automatic fallback

**Security Note**: API keys are stored in session state only and are never persisted to disk or committed to version control.

## Project Structure

```
golden68_framework/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── .gitignore                      # Git ignore rules
│
├── conf/
│   └── config.yaml                 # Configuration settings
│
├── data/
│   ├── dataset/
│   │   ├── golden68.json           # The 68 evaluation prompts
│   │   └── convert.py              # Dataset conversion utilities
│   ├── results/                    # Evaluation results (JSON)
│   ├── reports/                    # Generated reports (Markdown)
│   ├── audit/                      # Human audit records
│   ├── checkpoints/                # Smart resume checkpoints
│   └── cost_tracking/              # API usage logs
│
└── src/
    ├── __init__.py
    │
    ├── models/
    │   ├── __init__.py
    │   └── adapters.py             # LLM provider adapters
    │
    ├── judges/
    │   ├── __init__.py
    │   └── llm_judge.py            # LLM-as-Judge implementation
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── loader.py               # Dataset loader
    │   ├── scorer.py               # Scoring algorithms
    │   ├── comparison.py           # Multi-model comparison
    │   └── cost_tracker.py         # API cost tracking
    │
    ├── audit/
    │   ├── __init__.py
    │   └── human_audit.py          # Human verification system
    │
    ├── reporting/
    │   ├── __init__.py
    │   └── report_generator.py    # Report generation
    │
    └── api/
        └── api_server.py           # Optional REST API
```

## Usage Guide

### 1. Configure Your Evaluation

- Select the pillars to evaluate (Causality, Compliance, Consistency)
- Choose complexity levels (1-5)
- Set the number of prompts to test
- Configure API keys for judge and test models

### 2. Run Evaluation

- Click "Initialize & Start Evaluation"
- Monitor real-time progress
- View live cost tracking
- Checkpoints are saved automatically every 10 prompts

### 3. Review Results

- Examine LLM-Judge scores and reasoning
- View score distributions by pillar and level
- Identify failures and high-performing prompts
- Generate detailed AI analysis reports

### 4. Human Verification (Optional)

- Conduct expert audits on selected evaluations
- Provide human scores and reasoning
- Compare with LLM-Judge assessments
- Calculate Agreement Delta metrics

### 5. Final Report

- Generate comprehensive comparison reports
- Export results in Markdown format
- View regulatory readiness heatmaps
- Access historical evaluations via leaderboard

## Evaluation Metrics

### Scoring System

- **Score Range**: 1-10 for each prompt
- **Pass Threshold**: Typically 6/10 (configurable)
- **Overall Score**: Average across all evaluated prompts

### Agreement Delta

Measures correlation between LLM-Judge and human expert scores:

- **Delta Range**: 0.0 to 1.0
- **Interpretation**:
  - 0.9 - 1.0: Excellent agreement
  - 0.7 - 0.9: Good agreement
  - 0.5 - 0.7: Moderate agreement
  - Below 0.5: Poor agreement (judge recalibration needed)

### Weighted Leaderboard Score

`Weighted Score = (Average Score / 10) × 70% + (Pass Rate) × 30%`

This combines both score quality and consistency.

## Results & Reports

### Output Files

All evaluation results are saved in the `data/` directory:

- **Evaluation Results**: `data/results/llm_judge_YYYYMMDD_HHMMSS.json`
- **Final Reports**: `data/reports/final_report_YYYYMMDD_HHMMSS.md`
- **Human Audits**: `data/audit/audit_YYYYMMDD.json`

### Report Contents

Each comprehensive report includes:

1. Executive Summary
2. Overall Performance Metrics
3. Pillar-by-Pillar Analysis
4. Complexity Level Breakdown
5. Critical Issues Identified
6. Best Performances
7. Regulatory Readiness Heatmap
8. Recommendations for Improvement

## EU AI Act 2026 Compliance

The Compliance pillar directly maps to EU AI Act requirements:

- **Transparency Obligations**: Article 13 - Information disclosure
- **Safety & Fundamental Rights**: Article 9 - Risk management
- **Documentation Requirements**: Article 11 - Technical documentation
- **Risk Categorization**: Article 6 - High-risk AI systems classification

This framework helps organizations:
- Identify compliance gaps
- Document AI system capabilities
- Demonstrate due diligence
- Prepare for regulatory audits

## Contributing

We welcome contributions from the community! Here's how you can help:

### Reporting Issues

- Use GitHub Issues to report bugs
- Provide detailed reproduction steps
- Include system information and error messages

### Adding Features

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Expanding the Dataset

- Propose new prompts via Issues
- Ensure prompts map to one of the three pillars
- Assign appropriate complexity levels
- Include expected behaviors and EU Act references (if applicable)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **EU AI Act 2026**: Framework compliance mapping
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **LLM Providers**: Google, OpenAI, Anthropic, NVIDIA for API access

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{golden68_framework,
  title={Golden 68: AI Compliance \& Audit Framework},
  author={Avadhut Chakradeo},
  year={2026},
  url={https://github.com/Avdhoot-x7/golden68-ai-audit-framework}
}
```

---

**Developed as a Final Year Project**

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the maintainer.

---

**Version**: 2.0  
**Last Updated**: April 2026
