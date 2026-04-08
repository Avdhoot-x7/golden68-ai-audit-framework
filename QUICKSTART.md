# Golden 68 Framework - Quick Start Guide

Get up and running with the Golden 68 AI Audit Framework in 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- At least one LLM API key (see [API Keys](#getting-api-keys) below)

## Installation (3 Steps)

### Option 1: Automated Setup (Windows)

```bash
# 1. Run the setup script
setup.bat

# 2. Run the application
run.bat
```

### Option 2: Manual Setup (All Platforms)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

## Getting API Keys

You'll need API keys for the LLM providers you want to use:

### Recommended for Starting: Google Gemini (Free Tier Available)

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Copy your API key
4. **Free tier**: 15 requests/minute, 1M tokens/month

### Other Providers

- **OpenAI**: [Get Key](https://platform.openai.com/api-keys) - Paid only
- **Anthropic**: [Get Key](https://console.anthropic.com/) - $5 free credit
- **NVIDIA**: [Get Key](https://build.nvidia.com/) - Free credits available
- **OpenRouter**: [Get Key](https://openrouter.ai/keys) - Pay-as-you-go

## First Evaluation (5 Minutes)

### Step 1: Configure Your Setup

1. Open the app (it will open in your browser automatically)
2. In the **Judge LLM** section:
   - Select Provider: `gemini`
   - Model Name: `gemini-2.0-flash`
   - Paste your Gemini API key
3. In the **Testing LLM** section:
   - Select Provider: `gemini` (you can use the same)
   - Model Name: `gemini-2.0-flash`
   - Paste your API key

### Step 2: Select Evaluation Scope

1. **Pillar Selection**: Check all three boxes
   - Causality
   - Compliance
   - Consistency

2. **Complexity Levels**: Select levels 1-3 (recommended for first run)

3. **Max Prompts**: Set to `10` for a quick test

### Step 3: Run Evaluation

1. Click **"Initialize & Start Evaluation"**
2. Wait 2-5 minutes (depending on API speed)
3. Review results:
   - Overall Score
   - Pass Rate
   - Pillar breakdown
   - Individual prompt results

### Step 4: View Results

Navigate through the tabs:
- **LLM Judge Results**: See automated scoring
- **Detailed Log**: Expand each prompt to see model responses and judge reasoning
- **Charts**: View score distributions

## Understanding Your Results

### Score Interpretation

| Score Range | Grade | Meaning |
|-------------|-------|---------|
| 9.0 - 10.0 | A+ | Excellent performance |
| 8.0 - 8.9 | A | Strong performance |
| 7.0 - 7.9 | B | Good performance |
| 6.0 - 6.9 | C | Acceptable with improvements needed |
| 5.0 - 5.9 | D | Needs significant improvement |
| < 5.0 | F | Poor performance |

### Pass Rate

- **Pass**: Score ≥ 6.0
- **Fail**: Score < 6.0

Aim for at least 70% pass rate for production deployment.

## Next Steps

### Run Full Evaluation

1. Select all complexity levels (1-5)
2. Set max prompts to `68` (full dataset)
3. Run evaluation (takes 15-30 minutes)

### Human Audit (Optional)

1. After LLM-Judge evaluation completes
2. Click **"Continue to Human Verification"**
3. Review prompts and provide expert scores
4. Compare with LLM-Judge via Agreement Delta

### Model Comparison

1. Run evaluations with different models
2. Navigate to **Model Comparison** tab
3. View leaderboard and comparative analysis

## Common Issues

### Issue: API Key Error

**Error**: "Invalid API key" or "Authentication failed"

**Solution**: 
- Verify your API key is correct
- Check if you have credits/quota remaining
- Ensure you selected the correct provider

### Issue: Rate Limit Exceeded

**Error**: "Rate limit exceeded" or "429 Too Many Requests"

**Solution**:
- Wait 60 seconds and resume evaluation
- Add a backup API key in the configuration
- Use Smart Resume feature to continue from checkpoint

### Issue: Import Errors

**Error**: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
# Make sure virtual environment is activated
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Tips for Best Results

1. **Start Small**: Test with 10 prompts first to verify setup
2. **Use Free Tiers**: Gemini offers generous free tier for testing
3. **Set Backup Keys**: Configure backup API keys to avoid interruptions
4. **Monitor Costs**: Check the Cost Monitor dashboard regularly
5. **Save Results**: All evaluations are auto-saved to `data/results/`

## Sample Workflow

Here's a recommended workflow for your Final Year Project:

### Week 1: Setup & Initial Testing
- Install framework
- Run quick evaluation (10 prompts, Level 1-2)
- Familiarize yourself with the interface

### Week 2: Full Evaluation
- Run full 68-prompt evaluation
- Test with 2-3 different models
- Generate comparison reports

### Week 3: Human Audit
- Conduct human verification on 20-30 prompts
- Calculate Agreement Delta
- Identify judge calibration issues

### Week 4: Analysis & Reporting
- Export all results
- Generate comprehensive reports
- Create visualizations for presentation

## Getting Help

- **Documentation**: See full [README.md](README.md)
- **GitHub Issues**: Report bugs or request features
- **API Documentation**: Check provider docs for rate limits and pricing

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│        GOLDEN 68 QUICK REFERENCE        │
├─────────────────────────────────────────┤
│ Setup:                                  │
│   setup.bat (Windows)                   │
│   python -m venv venv && pip install -r │
│                                         │
│ Run:                                    │
│   streamlit run app.py                  │
│                                         │
│ Recommended First Run:                  │
│   • Pillars: All 3                      │
│   • Levels: 1-3                         │
│   • Prompts: 10                         │
│   • Time: ~5 minutes                    │
│                                         │
│ Full Evaluation:                        │
│   • Pillars: All 3                      │
│   • Levels: 1-5                         │
│   • Prompts: 68                         │
│   • Time: ~20-30 minutes                │
│                                         │
│ Results Location:                       │
│   data/results/*.json                   │
│   data/reports/*.md                     │
└─────────────────────────────────────────┘
```

---

**Ready to start?** Run `streamlit run app.py` and follow the steps above!

For detailed documentation, see [README.md](README.md)
