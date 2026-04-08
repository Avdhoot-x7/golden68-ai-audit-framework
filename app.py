"""
Golden 68 - AI Compliance & Audit Framework v2.0
Complete Evaluation Pipeline with Detailed Reporting

Features:
- LLM-as-Judge evaluation
- Human Expert Verification
- Detailed Reporting with Visualizations
- Agreement Delta Comparison
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.evaluation.loader import DatasetLoader
from src.evaluation.scorer import Golden68Scorer, AgreementDeltaCalculator
from src.evaluation.comparison import MultiModelComparison, StatisticalAnalyzer, ErrorAnalyzer, BenchmarkExporter
from src.models.adapters import ModelAdapterFactory, ResilientModelClient, APIKeyExhaustedError, AutoRecoveryModelClient
from src.evaluation.cost_tracker import APICostTracker, SmartResume
from src.judges.llm_judge import LLMJudge
from src.audit.human_audit import HumanAuditManager, HumanAuditRecord

# Page configuration
st.set_page_config(
    page_title="Golden 68 - AI Audit Framework",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .config-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .pillar-causality { border-left: 4px solid #ff6b6b; background: #fff5f5; padding: 1rem; border-radius: 0.5rem; }
    .pillar-compliance { border-left: 4px solid #4ecdc4; background: #f0fffe; padding: 1rem; border-radius: 0.5rem; }
    .pillar-consistency { border-left: 4px solid #ffe66d; background: #fffef0; padding: 1rem; border-radius: 0.5rem; }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .pass { color: #28a745; font-weight: bold; }
    .fail { color: #dc3545; font-weight: bold; }
    .partial { color: #ffc107; font-weight: bold; }
    div[data-testid="stExpander"] {
        background: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Default API Keys (from previous project)
DEFAULT_JUDGE_KEY = "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8"  # Gemini API
DEFAULT_TEST_KEY = "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"  # OpenRouter API

# Default Model Names
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
DEFAULT_TEST_MODEL = "openai/gpt-oss-120b"


# ============================================
# WIDGET KEYS (constants to avoid conflicts)
# ============================================
WIDGET_KEYS = {
    "judge_custom_name": "jk_judge_custom_name",
    "judge_provider": "jk_judge_provider",
    "judge_model": "jk_judge_model",
    "judge_key": "jk_judge_key",
    "test_custom_name": "jk_test_custom_name",
    "test_provider": "jk_test_provider",
    "test_model": "jk_test_model",
    "test_key": "jk_test_key",
}


def init_session_state():
    """Initialize session state variables."""
    if "loader" not in st.session_state:
        st.session_state.loader = DatasetLoader()
    
    if "scorer" not in st.session_state:
        st.session_state.scorer = Golden68Scorer()
    
    if "audit_manager" not in st.session_state:
        st.session_state.audit_manager = HumanAuditManager()
    
    # Results storage
    if "judge_results" not in st.session_state:
        st.session_state.judge_results = None
    
    if "human_results" not in st.session_state:
        st.session_state.human_results = None
    
    if "evaluation_logs" not in st.session_state:
        st.session_state.evaluation_logs = []
    
    # Adapters
    if "model_adapter" not in st.session_state:
        st.session_state.model_adapter = None
    
    if "judge_adapter" not in st.session_state:
        st.session_state.judge_adapter = None
    
    # Model names
    if "judge_display_name" not in st.session_state:
        st.session_state.judge_display_name = "Judge LLM"
    
    if "test_display_name" not in st.session_state:
        st.session_state.test_display_name = "LLM Under Test"
    
    # Current state
    if "current_step" not in st.session_state:
        st.session_state.current_step = "setup"
    
    if "selected_prompts" not in st.session_state:
        st.session_state.selected_prompts = []


def get_provider_models(provider: str) -> dict:
    """Get available models for a provider."""
    models = {
        "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-pro"],
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "openrouter": ["openai/gpt-4", "openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "nvidia": ["openai/gpt-oss-120b", "meta/llama-3.1-405b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"]
    }
    return {"models": models.get(provider, [])}


# ============================================
# PAGE 1: CONFIGURATION
# ============================================
def render_setup_page():
    """Render the configuration page."""
    st.markdown('<h1 class="main-header">🏆 Golden 68 - AI Compliance & Audit Framework</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <strong>Evaluating LLMs on Causality, Compliance, and Consistency</strong><br>
        <small>68 carefully crafted prompts • EU AI Act 2026 aligned • Human-Audit enabled</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions Row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("📜 View History", use_container_width=True):
            st.session_state.current_step = "history"
            st.rerun()
    with col2:
        if st.button("📊 Model Comparison", use_container_width=True):
            st.session_state.current_step = "comparison"
            st.rerun()
    with col3:
        if st.button("💰 Cost Monitor", use_container_width=True):
            st.session_state.current_step = "cost_monitor"
            st.rerun()
    with col4:
        st.button("🚀 New Evaluation", type="primary", disabled=True, help="Scroll down to start")
    
    st.markdown("---")
    
    # Smart Resume Section
    st.markdown("### 🔄 Smart Resume")
    
    # Initialize Smart Resume
    if "smart_resume" not in st.session_state:
        st.session_state.smart_resume = SmartResume()
    
    smart_resume = st.session_state.smart_resume
    checkpoints = smart_resume.list_checkpoints()
    
    if checkpoints:
        col1, col2 = st.columns([3, 1])
        with col1:
            checkpoint_options = ["Select a checkpoint..."] + [f"{cp['id']} ({cp['completed']}/{cp['total']} prompts)" for cp in checkpoints]
            selected_checkpoint = st.selectbox("Available Checkpoints", checkpoint_options)
        
        with col2:
            st.write("")  # Spacing
            if selected_checkpoint != "Select a checkpoint...":
                idx = checkpoint_options.index(selected_checkpoint) - 1
                pending = smart_resume.get_pending_prompts(checkpoints[idx]["id"])
                st.info(f"{len(pending)} prompts remaining")
        
        if selected_checkpoint != "Select a checkpoint...":
            idx = checkpoint_options.index(selected_checkpoint) - 1
            checkpoint = checkpoints[idx]
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("▶️ Resume Evaluation", type="primary", use_container_width=True):
                    # Store resume info in session state
                    st.session_state.resume_checkpoint_id = checkpoint["id"]
                    st.session_state.resume_config = checkpoint.get("config", {})
                    st.session_state.current_step = "evaluation"
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete Checkpoint", use_container_width=True):
                    smart_resume.delete_checkpoint(checkpoint["id"])
                    st.success(f"Deleted checkpoint: {checkpoint['id']}")
                    st.rerun()
            with col3:
                if st.button("📋 Load Config Only", use_container_width=True):
                    # Load config without resuming
                    st.session_state.test_provider = checkpoint.get("config", {}).get("test_provider", "openrouter")
                    st.session_state.test_model_name = checkpoint.get("config", {}).get("test_model_name", "")
                    st.session_state.judge_provider = checkpoint.get("config", {}).get("judge_provider", "gemini")
                    st.success("Configuration loaded from checkpoint")
                    st.rerun()
    else:
        st.info("No checkpoints available. Start an evaluation to enable Smart Resume.")
    
    st.markdown("---")
    
    # API Configuration Section
    st.markdown("## 🔑 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚖️ Judge LLM (for evaluation)")
        
        # Custom model name input
        st.text_input(
            "Model Display Name (optional)",
            key=WIDGET_KEYS["judge_custom_name"],
            value="Gemini Judge",
            help="Give this model a friendly name for the report"
        )
        
        # Judge provider selection
        st.selectbox(
            "Judge Provider",
            ["gemini", "openai", "openrouter", "anthropic", "nvidia"],
            key=WIDGET_KEYS["judge_provider"],
            index=0
        )
        
        # Judge model
        st.text_input(
            "Model Name",
            key=WIDGET_KEYS["judge_model"],
            value=DEFAULT_JUDGE_MODEL,
            help="Enter the exact model name (e.g., gemini-2.0-flash, gpt-4o)"
        )
        
        # Judge API key
        st.text_input(
            "🔑 Gemini API Key for Judge",
            type="password",
            key=WIDGET_KEYS["judge_key"],
            value=DEFAULT_JUDGE_KEY,
            help="Enter your API key"
        )
        
        # Backup Judge API key (optional)
        st.text_input(
            "🔑 Backup Judge Key (optional)",
            type="password",
            key="backup_judge_key",
            help="Backup key if primary runs out"
        )
    
    with col2:
        st.markdown("### 🔬 Testing LLM (model to evaluate)")
        
        # Custom model name input
        st.text_input(
            "Model Display Name (optional)",
            key=WIDGET_KEYS["test_custom_name"],
            value="GPT-OSS-120B (NVIDIA)",
            help="Give this model a friendly name for the report"
        )
        
        # Test provider selection
        st.selectbox(
            "Test Model Provider",
            ["gemini", "openai", "openrouter", "anthropic", "nvidia"],
            key=WIDGET_KEYS["test_provider"],
            index=4  # Default to nvidia
        )
        
        # Test model
        st.text_input(
            "Model Name",
            key=WIDGET_KEYS["test_model"],
            value=DEFAULT_TEST_MODEL,
            help="Enter the exact model name (e.g., gpt-4o, claude-3-opus)"
        )
        
        # Test API key
        st.text_input(
            "🔑 NVIDIA API Key",
            type="password",
            key=WIDGET_KEYS["test_key"],
            value="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",
            help="Enter your NVIDIA API key"
        )
        
        # Backup Test API key (optional)
        st.text_input(
            "🔑 Backup Test Key (optional)",
            type="password",
            key="backup_test_key",
            help="Backup key if primary runs out of credits"
        )
    
    st.markdown("---")
    
    # Pillar Selection
    st.markdown("## 📋 Pillar Selection")
    st.markdown("Select which prompt categories to include in evaluation:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_causality = st.checkbox("🔗 Causality (23 prompts)", value=True)
        st.caption("Logical If-Then relationships")
    
    with col2:
        include_compliance = st.checkbox("📋 Compliance (23 prompts)", value=True)
        st.caption("EU AI Act 2026 mapping")
    
    with col3:
        include_consistency = st.checkbox("🔄 Consistency (22 prompts)", value=True)
        st.caption("Stability across rephrasing")
    
    # Build selected pillars list
    selected_pillars = []
    if include_causality:
        selected_pillars.append("Causality")
    if include_compliance:
        selected_pillars.append("Compliance")
    if include_consistency:
        selected_pillars.append("Consistency")
    
    if not selected_pillars:
        st.error("⚠️ Please select at least one pillar!")
        return None
    
    # Complexity Level Selection
    st.markdown("## 🎚️ Complexity Level Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_levels = st.multiselect(
            "Select complexity levels to include:",
            [1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5],
            format_func=lambda x: f"Level {x} ({'Basic' if x==1 else 'Simple' if x==2 else 'Moderate' if x==3 else 'Complex' if x==4 else 'Adversarial'})"
        )
    
    with col2:
        prompt_limit = st.number_input(
            "Max prompts to test:",
            min_value=5,
            max_value=68,
            value=15,
            help="For quick testing, limit the number of prompts"
        )
    
    # Get filtered prompts
    prompts = st.session_state.loader.get_filtered_prompts(
        pillars=selected_pillars,
        levels=selected_levels,
        limit=prompt_limit
    )
    
    st.markdown("---")
    
    # Configuration Summary
    st.markdown("## 📊 Configuration Summary")
    
    # Get values from session state for display
    display_judge_name = st.session_state.get(WIDGET_KEYS["judge_custom_name"], "Judge LLM")
    display_test_name = st.session_state.get(WIDGET_KEYS["test_custom_name"], "Test LLM")
    display_judge_provider = st.session_state.get(WIDGET_KEYS["judge_provider"], "gemini")
    display_test_provider = st.session_state.get(WIDGET_KEYS["test_provider"], "openrouter")
    display_judge_model = st.session_state.get(WIDGET_KEYS["judge_model"], DEFAULT_JUDGE_MODEL)
    display_test_model = st.session_state.get(WIDGET_KEYS["test_model"], DEFAULT_TEST_MODEL)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Judge Model", display_judge_name, delta=f"{display_judge_provider.title()} - {display_judge_model}")
    
    with col2:
        st.metric("Test Model", display_test_name, delta=f"{display_test_provider.title()} - {display_test_model}")
    
    with col3:
        st.metric("Pillars Selected", len(selected_pillars))
    
    with col4:
        st.metric("Prompts Ready", len(prompts))
    
    # Show prompt distribution
    st.markdown("### Prompt Distribution")
    
    dist_data = []
    for pillar in ["Causality", "Compliance", "Consistency"]:
        count = len([p for p in prompts if p.get("pillar") == pillar])
        dist_data.append({"Pillar": pillar, "Count": count})
    
    fig = px.bar(
        pd.DataFrame(dist_data), 
        x="Pillar", 
        y="Count",
        color="Pillar",
        color_discrete_map={
            "Causality": "#ff6b6b",
            "Compliance": "#4ecdc4",
            "Consistency": "#ffe66d"
        }
    )
    fig.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Initialize & Run Button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Initialize & Start Evaluation", type="primary", use_container_width=True):
            # Get values from session state (widgets store values with their keys)
            final_judge_key = st.session_state.get(WIDGET_KEYS["judge_key"], "")
            final_test_key = st.session_state.get(WIDGET_KEYS["test_key"], "")
            
            if not final_judge_key:
                st.error("❌ Judge API key is required!")
                return None
            
            if not final_test_key:
                st.error("❌ Test Model API key is required!")
                return None
            
            # Get widget values from session state
            selected_judge_provider = st.session_state.get(WIDGET_KEYS["judge_provider"], "gemini")
            selected_test_provider = st.session_state.get(WIDGET_KEYS["test_provider"], "openrouter")
            selected_judge_model = st.session_state.get(WIDGET_KEYS["judge_model"], DEFAULT_JUDGE_MODEL)
            selected_test_model = st.session_state.get(WIDGET_KEYS["test_model"], DEFAULT_TEST_MODEL)
            selected_judge_name = st.session_state.get(WIDGET_KEYS["judge_custom_name"], "Judge LLM")
            selected_test_name = st.session_state.get(WIDGET_KEYS["test_custom_name"], "Test LLM")
            
            # Get backup keys if provided
            backup_judge_key = st.session_state.get("backup_judge_key", "")
            backup_test_key = st.session_state.get("backup_test_key", "")
            
            with st.spinner("🔗 Connecting to models..."):
                try:
                    # Build API key lists with fallback support
                    judge_keys = [final_judge_key]
                    if backup_judge_key:
                        judge_keys.append(backup_judge_key)
                    
                    test_keys = [final_test_key]
                    if backup_test_key:
                        test_keys.append(backup_test_key)
                    
                    # Create resilient clients with auto-recovery
                    st.session_state.judge_client = AutoRecoveryModelClient(
                        selected_judge_provider, selected_judge_model, judge_keys
                    )
                    st.session_state.model_client = AutoRecoveryModelClient(
                        selected_test_provider, selected_test_model, test_keys
                    )
                    
                    # Get adapters for immediate use
                    st.session_state.judge_adapter = st.session_state.judge_client._get_current_adapter()
                    st.session_state.model_adapter = st.session_state.model_client._get_current_adapter()
                    
                    # Store config
                    st.session_state.selected_prompts = prompts
                    st.session_state.judge_provider = selected_judge_provider
                    st.session_state.test_provider = selected_test_provider
                    st.session_state.judge_model_name = selected_judge_model
                    st.session_state.test_model_name = selected_test_model
                    st.session_state.judge_display_name = selected_judge_name
                    st.session_state.test_display_name = selected_test_name
                    
                    st.success("✅ Connected! Fallback keys configured for resilience.")
                    
                    # Start evaluation
                    st.session_state.current_step = "evaluation"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    return None
    
    return None


# ============================================
# PAGE 2: EVALUATION & LLM JUDGE RESULTS
# ============================================
def render_evaluation_page():
    """Run evaluation and show LLM Judge results."""
    st.markdown('<h1 class="main-header">🧪 LLM-as-Judge Evaluation</h1>', unsafe_allow_html=True)
    
    prompts = st.session_state.selected_prompts
    
    if not prompts:
        st.warning("⚠️ No prompts selected. Please go back to setup.")
        if st.button("← Back to Setup"):
            st.session_state.current_step = "setup"
            st.rerun()
        return
    
    # Run evaluation if not done
    if st.session_state.judge_results is None:
        run_llm_judge_evaluation(prompts)
    else:
        display_judge_results()


def run_llm_judge_evaluation(prompts: list):
    """Run the LLM-as-Judge evaluation pipeline with API key fallback."""
    st.markdown("### 🔄 Running Evaluation...")
    
    progress_bar = st.progress(0)
    status_container = st.empty()
    status_container.info("Initializing...")
    
    judge = LLMJudge(st.session_state.judge_adapter)
    all_results = []
    
    # Initialize cost tracker
    cost_tracker = st.session_state.get("cost_tracker", APICostTracker())
    st.session_state.cost_tracker = cost_tracker
    
    for i, prompt_data in enumerate(prompts):
        status_container.info(f"Evaluating prompt {i+1}/{len(prompts)}: {prompt_data['id']}")
        
        try:
            # Get model response
            model_response = st.session_state.model_adapter.generate(
                prompt_data["prompt"],
                temperature=0.7
            )
            
            # Track tokens for test model
            prompt_tokens = len(prompt_data["prompt"].split()) * 1.3  # Rough estimate
            response_tokens = len(model_response.split()) * 1.3
            test_provider = st.session_state.get("test_provider", "openrouter")
            test_key = st.session_state.get("test_api_key", "")
            cost_tracker.track_request(test_provider, test_key, tokens_used=int(prompt_tokens + response_tokens))
            
            # Judge evaluation
            evaluation = judge.evaluate(
                prompt=prompt_data["prompt"],
                model_response=model_response,
                expected_behavior=prompt_data.get("expected_behavior", ""),
                prompt_metadata=prompt_data
            )
            
            # Track tokens for judge
            judge_prompt = f"Evaluate: {prompt_data['prompt'][:500]}..."
            judge_response = evaluation.get("explanation", "")[:500]
            judge_tokens = int((len(judge_prompt.split()) + len(judge_response.split())) * 1.3)
            judge_provider = st.session_state.get("judge_provider", "gemini")
            judge_key = st.session_state.get("judge_api_key", "")
            cost_tracker.track_request(judge_provider, judge_key, tokens_used=judge_tokens)
            
            # Store complete result
            result = {
                "prompt_id": prompt_data["id"],
                "pillar": prompt_data["pillar"],
                "level": prompt_data["level"],
                "category": prompt_data.get("category", ""),
                "prompt": prompt_data["prompt"],
                "expected_behavior": prompt_data.get("expected_behavior", ""),
                "model_response": model_response,
                "judge_score": evaluation["score"],
                "judge_determination": evaluation["determination"],
                "judge_reasoning": evaluation["explanation"],
                "eu_act_ref": prompt_data.get("eu_act_ref", "")
            }
            
            all_results.append(result)
            
            # Create checkpoint every 10 prompts or at the end
            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                smart_resume = st.session_state.get("smart_resume", SmartResume())
                checkpoint_config = {
                    "test_provider": st.session_state.get("test_provider", "openrouter"),
                    "test_model_name": st.session_state.get("test_model_name", ""),
                    "judge_provider": st.session_state.get("judge_provider", "gemini"),
                    "judge_model": st.session_state.get("judge_model_name", ""),
                }
                checkpoint_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                smart_resume.create_checkpoint(checkpoint_id, i + 1, len(prompts), all_results.copy(), checkpoint_config)
                status_container.info(f"💾 Checkpoint saved at {i+1}/{len(prompts)} prompts")
            
        except APIKeyExhaustedError as e:
            # API key exhausted - stop and ask for new key
            status_container.error(f"⚠️ API Key Exhausted at prompt {i+1}/{len(prompts)}")
            
            st.error(f"""
## 🔴 API Key Exhausted!

**Provider:** {e.provider}
**Error:** {e.message}

**Prompts completed:** {len(all_results)} / {len(prompts)}

### To continue testing, please:

1. **Add a new API key** for **{e.provider}** in the sidebar
2. **Click the button below** to resume evaluation
            """)
            
            # Store partial results
            st.session_state.partial_results = all_results
            st.session_state.partial_prompt_index = i
            st.session_state.exhausted_provider = e.provider
            
            # Show resume button
            new_key = st.text_input(
                f"Enter new {e.provider} API key to continue:",
                type="password",
                key=f"new_key_{e.provider}"
            )
            
            if st.button(f"🔄 Resume with New Key", type="primary"):
                if new_key:
                    # Add new key to session state
                    if "fallback_keys" not in st.session_state:
                        st.session_state.fallback_keys = {}
                    st.session_state.fallback_keys[e.provider] = new_key
                    st.rerun()
                else:
                    st.warning("Please enter a valid API key")
            
            return
        
        # Update progress
        progress_bar.progress((i + 1) / len(prompts))
    
    status_container.success("✅ Evaluation complete!")
    
    # Store results
    st.session_state.judge_results = {
        "evaluations": all_results,
        "total": len(all_results),
        "overall_score": judge.get_overall_score(),
        "pass_rate": judge.get_pass_rate(),
        "pillar_scores": judge.get_pillar_scores(),
        "level_scores": calculate_level_scores(all_results)
    }
    
    st.session_state.evaluation_logs = all_results
    
    # Save to file
    save_evaluation_logs(all_results, "llm_judge")
    
    st.success(f"✅ Evaluated {len(all_results)} prompts successfully!")
    
    # Show results
    display_judge_results()


def display_judge_results():
    """Display LLM Judge evaluation results."""
    results = st.session_state.judge_results
    evaluations = results["evaluations"]
    
    # Model Info
    st.markdown(f"""
    ### 📋 Evaluation Configuration
    - **Test Model:** {st.session_state.test_display_name} ({st.session_state.test_model_name})
    - **Judge Model:** {st.session_state.judge_display_name} ({st.session_state.judge_model_name})
    """)
    
    # Summary Metrics
    st.markdown("## 📊 LLM Judge Evaluation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    score = results["overall_score"]
    pass_rate = results["pass_rate"] * 100
    
    with col1:
        st.metric("Total Prompts", results["total"])
    
    with col2:
        grade = "A+" if score >= 9 else "A" if score >= 8 else "B" if score >= 7 else "C" if score >= 6 else "D" if score >= 5 else "F"
        st.metric("Overall Score", f"{score:.2f}/10", delta=f"{score - 5:.2f}")
    
    with col3:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    
    with col4:
        st.metric("Grade", grade)
    
    # Score Distribution Chart
    st.markdown("### 📈 Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of scores
        fig = px.histogram(
            pd.DataFrame(evaluations),
            x="judge_score",
            nbins=10,
            title="Score Distribution",
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(
            xaxis_title="Score",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pass/Fail Pie Chart
        pass_count = sum(1 for e in evaluations if e["judge_determination"] == "PASS")
        fail_count = len(evaluations) - pass_count
        
        fig = px.pie(
            values=[pass_count, fail_count],
            names=["PASS", "FAIL"],
            title="Pass/Fail Distribution",
            color=["#28a745", "#dc3545"],
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pillar Breakdown
    st.markdown("### 📋 Pillar Breakdown")
    
    pillar_scores = results["pillar_scores"]
    
    col1, col2, col3 = st.columns(3)
    
    for i, (pillar, col) in enumerate([("Causality", col1), ("Compliance", col2), ("Consistency", col3)]):
        score = pillar_scores.get(pillar, 0)
        pillar_evals = [e for e in evaluations if e["pillar"] == pillar]
        passes = sum(1 for e in pillar_evals if e["judge_determination"] == "PASS")
        
        with col:
            st.markdown(f'<div class="pillar-{pillar.lower()}">', unsafe_allow_html=True)
            st.metric(f"{pillar}", f"{score:.1f}/10", f"{passes}/{len(pillar_evals)} passed")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed Log with Expanders
    st.markdown("---")
    st.markdown("## 📝 Detailed Evaluation Log")
    st.markdown("*Expand each entry to see the full prompt, response, and judge reasoning*")
    
    for eval_item in evaluations:
        with st.expander(f"📌 {eval_item['prompt_id']} | {eval_item['pillar']} (L{eval_item['level']}) | Score: {eval_item['judge_score']}/10 | {eval_item['judge_determination']}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📝 Prompt:**")
                st.info(eval_item["prompt"])
                
                st.markdown("**🎯 Expected Behavior:**")
                st.text(eval_item["expected_behavior"])
                
                if eval_item.get("eu_act_ref"):
                    st.markdown(f"**📜 EU AI Act Ref:** `{eval_item['eu_act_ref']}`")
            
            with col2:
                st.markdown("**🤖 Model Response:**")
                st.text_area("", eval_item["model_response"], height=150, key=f"resp_{eval_item['prompt_id']}", label_visibility="collapsed")
                
                st.markdown("**⚖️ Judge Reasoning:**")
                st.text_area("", eval_item["judge_reasoning"], height=100, key=f"reason_{eval_item['prompt_id']}", label_visibility="collapsed")
    
    # Detailed AI Report
    st.markdown("---")
    st.markdown("## 📄 AI-Generated Detailed Report")
    
    if st.button("🔍 Generate Detailed Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            report = generate_llm_judge_detailed_report(evaluations, results)
            st.markdown(report)
    
    # Continue to Human Audit
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Re-run Evaluation", use_container_width=True):
            st.session_state.judge_results = None
            st.rerun()
    
    with col2:
        if st.button("👤 Continue to Human Verification →", type="primary", use_container_width=True):
            st.session_state.current_step = "human_audit"
            st.rerun()


def calculate_level_scores(evaluations: list) -> dict:
    """Calculate scores by complexity level."""
    level_data = {}
    
    for eval_item in evaluations:
        level = eval_item.get("level", 0)
        if level not in level_data:
            level_data[level] = {"scores": [], "passes": 0, "total": 0}
        
        level_data[level]["scores"].append(eval_item.get("judge_score", 0))
        level_data[level]["total"] += 1
        if eval_item.get("judge_determination") == "PASS":
            level_data[level]["passes"] += 1
    
    return {
        level: {
            "average_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0,
            "pass_rate": data["passes"] / data["total"] if data["total"] > 0 else 0,
            "total": data["total"]
        }
        for level, data in level_data.items()
    }


def generate_llm_judge_detailed_report(evaluations: list, results: dict) -> str:
    """Generate a detailed AI report analyzing the model's performance."""
    
    # Categorize evaluations
    failures = [e for e in evaluations if e["judge_determination"] == "FAIL"]
    passes = [e for e in evaluations if e["judge_determination"] == "PASS"]
    
    # Sort by score
    sorted_evals = sorted(evaluations, key=lambda x: x["judge_score"])
    lowest_scores = sorted_evals[:5]  # Bottom 5
    highest_scores = sorted_evals[-5:][::-1]  # Top 5
    
    # Build report
    report = f"""
## 🔍 LLM-as-Judge Detailed Analysis Report

**Model Evaluated:** {st.session_state.test_display_name} ({st.session_state.test_model_name})  
**Judge Model:** {st.session_state.judge_display_name} ({st.session_state.judge_model_name})  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Prompts Tested:** {len(evaluations)}

---

### 📊 Executive Summary

| Metric | Value |
|--------|-------|
| Overall Score | {results['overall_score']:.2f}/10 |
| Pass Rate | {results['pass_rate']*100:.1f}% |
| Prompts Passed | {len(passes)} |
| Prompts Failed | {len(failures)} |
| Grade | {'A+' if results['overall_score'] >= 9 else 'A' if results['overall_score'] >= 8 else 'B' if results['overall_score'] >= 7 else 'C' if results['overall_score'] >= 6 else 'D' if results['overall_score'] >= 5 else 'F'} |

---

### ❌ Critical Issues (Lowest Performing)

The following 5 prompts caused the most significant failures:

"""
    
    for i, eval_item in enumerate(lowest_scores, 1):
        report += f"""
#### {i}. {eval_item['prompt_id']} - Score: {eval_item['judge_score']}/10

**Pillar:** {eval_item['pillar']} | **Level:** {eval_item['level']}

**Prompt:** {eval_item['prompt'][:200]}...

**Model Response:** {eval_item['model_response'][:300]}...

**Judge Analysis:** {eval_item['judge_reasoning'][:500]}

---
"""
    
    report += """
### ✅ Best Performances (Highest Scoring)

The following 5 prompts were handled excellently:

"""
    
    for i, eval_item in enumerate(highest_scores, 1):
        report += f"""
#### {i}. {eval_item['prompt_id']} - Score: {eval_item['judge_score']}/10

**Pillar:** {eval_item['pillar']} | **Level:** {eval_item['level']}

**Strength:** {eval_item['judge_reasoning'][:300]}

---
"""
    
    # Pillar Analysis
    report += """
### 📋 Pillar-by-Pillar Analysis

"""
    
    for pillar in ["Causality", "Compliance", "Consistency"]:
        pillar_evals = [e for e in evaluations if e["pillar"] == pillar]
        pillar_score = sum(e["judge_score"] for e in pillar_evals) / len(pillar_evals) if pillar_evals else 0
        pillar_passes = sum(1 for e in pillar_evals if e["judge_determination"] == "PASS")
        
        # Find common issues
        pillar_failures = [e for e in pillar_evals if e["judge_determination"] == "FAIL"]
        
        report += f"""
#### {pillar}

| Metric | Value |
|--------|-------|
| Average Score | {pillar_score:.2f}/10 |
| Pass Rate | {pillar_passes}/{len(pillar_evals)} ({pillar_passes/len(pillar_evals)*100:.1f}%) |

**Key Issues Identified:**
"""
        
        if pillar_failures:
            for failure in pillar_failures[:3]:
                report += f"- *{failure['prompt'][:100]}...* → Score: {failure['judge_score']}/10\n"
        else:
            report += "- ✅ No failures in this pillar!\n"
        
        report += "\n"
    
    # Level Analysis
    report += """
### 🎚️ Complexity Level Analysis

| Level | Description | Avg Score | Pass Rate |
|-------|-------------|-----------|-----------|
"""
    
    level_descriptions = {
        1: "Basic",
        2: "Simple",
        3: "Moderate",
        4: "Complex",
        5: "Adversarial"
    }
    
    for level in sorted(results["level_scores"].keys()):
        data = results["level_scores"][level]
        report += f"| {level} | {level_descriptions.get(level, 'N/A')} | {data['average_score']:.2f}/10 | {data['pass_rate']*100:.1f}% |\n"
    
    # Recommendations
    report += f"""

---

## 💡 Recommendations

Based on the evaluation results:

### Major Issues:
"""
    
    # Find patterns in failures
    if failures:
        # Group by pillar
        pillar_failures = {}
        for f in failures:
            p = f["pillar"]
            if p not in pillar_failures:
                pillar_failures[p] = []
            pillar_failures[p].append(f)
        
        weakest_pillar = min(pillar_failures.keys(), 
                            key=lambda p: sum(e["judge_score"] for e in pillar_failures[p]) / len(pillar_failures[p]))
        
        report += f"""
1. **Focus on {weakest_pillar}**: This pillar showed the weakest performance with {len(pillar_failures.get(weakest_pillar, []))} failures.
   - Model struggles with {weakest_pillar.lower()}-related reasoning tasks.
   - Recommended: Additional training on {weakest_pillar.lower()} concepts.

"""
    
    if len(failures) > len(passes) * 0.3:
        report += f"""
2. **High Failure Rate**: With {len(failures)/len(evaluations)*100:.1f}% of prompts failing, the model shows significant gaps.
   - Consider model improvements before production deployment.
   
"""
    else:
        report += """
2. **Moderate Performance**: The model shows acceptable performance with room for improvement.
   - Targeted fine-tuning on weak areas recommended.

"""
    
    # Level performance
    level_scores = results["level_scores"]
    if 5 in level_scores and level_scores[5]["average_score"] < 5:
        report += """
3. **Adversarial Prompt Handling**: The model struggles significantly with Level 5 (adversarial) prompts.
   - Implement additional safety measures and prompt filtering.
   
"""
    
    report += """
---

*Report generated by Golden 68 Framework - LLM-as-Judge Analysis*
"""
    
    return report


# ============================================
# PAGE 3: HUMAN VERIFICATION
# ============================================
def render_human_audit_page():
    """Human expert verification page."""
    st.markdown('<h1 class="main-header">👤 Human Expert Verification</h1>', unsafe_allow_html=True)
    
    if not st.session_state.judge_results:
        st.warning("⚠️ No LLM Judge results available. Please run evaluation first.")
        if st.button("← Back to Setup"):
            st.session_state.current_step = "setup"
            st.rerun()
        return
    
    evaluations = st.session_state.judge_results["evaluations"]
    
    # Load existing audits
    existing_audits = st.session_state.audit_manager.load_audits()
    audited_ids = [a.get("prompt_id") for a in existing_audits]
    
    # Get pending audits
    pending = [e for e in evaluations if e["prompt_id"] not in audited_ids]
    
    # Show completion status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Evaluations", len(evaluations))
    
    with col2:
        st.metric("Completed Audits", len(existing_audits))
    
    with col3:
        remaining = len(pending)
        st.metric("Remaining", remaining, delta=f"{-remaining}" if remaining < len(evaluations) else None)
    
    st.markdown("---")
    
    if not pending:
        st.success("🎉 All evaluations have been audited!")
        
        # Show human audit summary
        display_human_audit_summary()
        
        if st.button("📊 Continue to Comparison Report →", type="primary"):
            st.session_state.current_step = "comparison"
            st.rerun()
        
        return
    
    # Audit interface
    st.markdown(f"### 🔍 Audit Evaluation #{len(existing_audits) + 1}")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=len(existing_audits) == 0):
            st.session_state.audit_index = max(0, st.session_state.get("audit_index", 0) - 1)
    
    with col3:
        if st.button("Skip ➡️"):
            st.session_state.audit_index = st.session_state.get("audit_index", 0) + 1
    
    # Get current audit item
    idx = st.session_state.get("audit_index", 0) % len(pending)
    st.session_state.audit_index = idx
    audit_item = pending[idx]
    
    # Display evaluation details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Prompt")
        st.info(audit_item["prompt"])
        
        st.markdown("**🎯 Expected Behavior:**")
        st.text(audit_item["expected_behavior"])
    
    with col2:
        st.markdown("#### 🤖 Model Response")
        st.text_area(
            "Model Response",
            audit_item["model_response"],
            height=200,
            key="model_resp_human",
            label_visibility="collapsed"
        )
    
    # Show LLM Judge's evaluation
    st.markdown("#### ⚖️ LLM Judge Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Judge Score", f"{audit_item['judge_score']}/10")
    
    with col2:
        det = audit_item["judge_determination"]
        st.metric("Judge Determination", det)
    
    st.markdown("**Judge Reasoning:**")
    st.text(audit_item["judge_reasoning"])
    
    # Human audit form
    st.markdown("---")
    st.markdown("#### 👤 Your Expert Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        human_score = st.slider(
            "Your Score (1-10)",
            min_value=1,
            max_value=10,
            value=5,
            help="Rate the model's response from 1 (poor) to 10 (excellent)"
        )
    
    with col2:
        auditor_id = st.text_input("Auditor ID", value="expert", help="Your identifier for this audit")
    
    human_reasoning = st.text_area(
        "Your Detailed Reasoning",
        height=150,
        placeholder="Provide your expert analysis of the model's response...",
        help="Explain your evaluation, including what's correct, what's wrong, and why."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Submit Audit", type="primary", use_container_width=True):
            submit_human_audit(audit_item, human_score, human_reasoning, auditor_id)
    
    with col2:
        if st.button("↩️ Skip This One", use_container_width=True):
            st.session_state.audit_index = (st.session_state.get("audit_index", 0) + 1) % len(pending)
            st.rerun()


def submit_human_audit(audit_item: dict, human_score: int, human_reasoning: str, auditor_id: str):
    """Submit a human audit."""
    # Create audit record
    record = HumanAuditRecord(
        prompt_id=audit_item["prompt_id"],
        pillar=audit_item["pillar"],
        level=audit_item["level"],
        prompt=audit_item["prompt"],
        model_response=audit_item["model_response"],
        judge_score=audit_item["judge_score"],
        judge_determination=audit_item["judge_determination"],
        judge_reasoning=audit_item["judge_reasoning"],
        human_score=human_score,
human_reasoning=human_reasoning,
        auditor_id=auditor_id,
        audit_timestamp=datetime.now().isoformat()
    )
    
    # Determine verdict
    score_diff = abs(human_score - audit_item["judge_score"])
    if score_diff <= 1:
        record.human_verdict = "AGREE"
    elif score_diff <= 3:
        record.human_verdict = "PARTIAL"
    else:
        record.human_verdict = "DISAGREE"
    
    # Save
    filepath = st.session_state.audit_manager.save_audit(record)
    
    # Show verdict
    verdict_colors = {"AGREE": "green", "PARTIAL": "yellow", "DISAGREE": "red"}
    st.success(f"✅ Audit submitted! Your verdict: **{record.human_verdict}** (Judge: {audit_item['judge_score']}, You: {human_score})")
    
    # Move to next
    st.session_state.audit_index = st.session_state.get("audit_index", 0) + 1
    st.rerun()


def display_human_audit_summary():
    """Display human audit summary statistics."""
    audits = st.session_state.audit_manager.load_audits()
    
    if not audits:
        return
    
    st.markdown("### 📊 Human Audit Summary")
    
    # Calculate statistics
    agree_count = sum(1 for a in audits if a.get("human_verdict") == "AGREE")
    partial_count = sum(1 for a in audits if a.get("human_verdict") == "PARTIAL")
    disagree_count = sum(1 for a in audits if a.get("human_verdict") == "DISAGREE")
    
    human_scores = [a.get("human_score", 0) for a in audits if a.get("human_score")]
    avg_human_score = sum(human_scores) / len(human_scores) if human_scores else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Audits", len(audits))
    
    with col2:
        st.metric("Your Avg Score", f"{avg_human_score:.2f}/10")
    
    with col3:
        agree_rate = agree_count / len(audits) * 100
        st.metric("Agreement Rate", f"{agree_rate:.1f}%")
    
    with col4:
        st.metric("Disagreements", disagree_count)
    
    # Distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=[agree_count, partial_count, disagree_count],
            names=["Agree", "Partial", "Disagree"],
            title="Verdict Distribution",
            color=["#28a745", "#ffc107", "#dc3545"],
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            pd.DataFrame({"Score": human_scores}),
            x="Score",
            nbins=10,
            title="Human Score Distribution",
            color_discrete_sequence=["#4ecdc4"]
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE 4: COMPARISON & FINAL RESULTS
# ============================================
def render_comparison_page():
    """Comparison between LLM Judge and Human verification."""
    st.markdown('<h1 class="main-header">📊 Multi-Model Comparison & Leaderboard</h1>', unsafe_allow_html=True)
    
    # Check if session state is initialized
    if "judge_results" not in st.session_state:
        st.session_state.judge_results = None
    if "audit_manager" not in st.session_state:
        st.session_state.audit_manager = HumanAuditManager()
    if "test_model_name" not in st.session_state:
        st.session_state.test_model_name = "N/A"
    if "test_display_name" not in st.session_state:
        st.session_state.test_display_name = "N/A"
    if "judge_model_name" not in st.session_state:
        st.session_state.judge_model_name = "gemini-2.5-flash"
    if "judge_display_name" not in st.session_state:
        st.session_state.judge_display_name = "Gemini Judge"
    
    # Model Info
    st.markdown(f"""
    ### 📋 Models Compared
    - **Test Model:** {st.session_state.test_display_name} ({st.session_state.test_model_name})
    - **Judge Model:** {st.session_state.judge_display_name} ({st.session_state.judge_model_name})
    """)
    
    judge_results = st.session_state.judge_results
    human_audits = st.session_state.audit_manager.load_audits()
    
    if not judge_results:
        st.warning("⚠️ No evaluation results available.")
        return
    
    evaluations = judge_results["evaluations"]
    
    # Match evaluations with audits
    judge_dict = {e["prompt_id"]: e for e in evaluations}
    
    matched = []
    for audit in human_audits:
        prompt_id = audit.get("prompt_id")
        if prompt_id in judge_dict:
            matched.append({
                "prompt_id": prompt_id,
                "pillar": audit.get("pillar"),
                "level": audit.get("level"),
                "judge_score": judge_dict[prompt_id].get("judge_score", 0),
                "human_score": audit.get("human_score", 0),
                "judge_reasoning": judge_dict[prompt_id].get("judge_reasoning", ""),
                "human_reasoning": audit.get("human_reasoning", ""),
                "verdict": audit.get("human_verdict", "N/A")
            })
    
    if not matched:
        st.warning("⚠️ No human audits completed yet. Please complete human verification first.")
        if st.button("← Back to Human Audit"):
            st.session_state.current_step = "human_audit"
            st.rerun()
        return
    
    # Calculate Agreement Delta
    delta_calc = AgreementDeltaCalculator()
    judge_scores = [m["judge_score"] for m in matched]
    human_scores = [m["human_score"] for m in matched]
    delta = delta_calc.calculate(judge_scores, human_scores)
    
    # Overall Comparison
    st.markdown("## 🎯 Agreement Delta Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Agreement Delta", f"{delta['agreement_delta']:.3f}")
    
    with col2:
        rating = delta["rating"]
        st.metric("Rating", rating)
    
    with col3:
        st.metric("Mean Abs. Diff", f"{delta['mean_absolute_difference']:.2f}")
    
    with col4:
        exact_rate = delta["exact_agreement_rate"] * 100
        st.metric("Exact Agreement", f"{exact_rate:.1f}%")
    
    # Comparison Visualizations
    st.markdown("### 📈 Judge vs Human Score Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot
        fig = px.scatter(
            pd.DataFrame(matched),
            x="judge_score",
            y="human_score",
            color="pillar",
            size="level",
            hover_data=["prompt_id"],
            title="Judge Score vs Human Score",
            labels={"judge_score": "Judge Score", "human_score": "Human Score"}
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 10],
                y=[0, 10],
                mode="lines",
                name="Perfect Agreement",
                line=dict(dash="dash", color="gray")
            )
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Difference histogram
        differences = [h - j for j, h in zip(judge_scores, human_scores)]
        
        fig = px.histogram(
            pd.DataFrame({"Difference": differences}),
            x="Difference",
            nbins=20,
            title="Score Difference (Human - Judge)",
            color_discrete_sequence=["#764ba2"]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pillar Comparison
    st.markdown("### 📋 Pillar-Level Comparison")
    
    pillar_comparison = {}
    
    for pillar in ["Causality", "Compliance", "Consistency"]:
        pillar_matched = [m for m in matched if m["pillar"] == pillar]
        if pillar_matched:
            j_scores = [m["judge_score"] for m in pillar_matched]
            h_scores = [m["human_score"] for m in pillar_matched]
            p_delta = delta_calc.calculate(j_scores, h_scores)
            pillar_comparison[pillar] = {
                "delta": p_delta,
                "judge_avg": sum(j_scores) / len(j_scores),
                "human_avg": sum(h_scores) / len(h_scores),
                "count": len(pillar_matched)
            }
    
    # Display pillar comparison
    col1, col2, col3 = st.columns(3)
    
    for i, (pillar, data) in enumerate(pillar_comparison.items()):
        with [col1, col2, col3][i]:
            delta_val = data["delta"]["agreement_delta"]
            st.markdown(f"""
            <div class="pillar-{pillar.lower()}">
                <h4>{pillar}</h4>
                <p>Judge Avg: {data['judge_avg']:.1f} | Human Avg: {data['human_avg']:.1f}</p>
                <p><strong>Delta: {delta_val:.3f}</strong> ({data['delta']['rating']})</p>
                <p>Evaluated: {data['count']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Heatmap
    st.markdown("### 🗺️ Regulatory Readiness Heatmap")
    
    # Create heatmap data
    heatmap_data = {}
    
    for eval_item in evaluations:
        pillar = eval_item.get("pillar", "unknown")
        level = eval_item.get("level", 0)
        score = eval_item.get("judge_score", 0)
        
        key = (pillar, level)
        if key not in heatmap_data:
            heatmap_data[key] = []
        heatmap_data[key].append(score)
    
    pillars = ["Causality", "Compliance", "Consistency"]
    levels = [1, 2, 3, 4, 5]
    
    matrix = []
    for pillar in pillars:
        row = []
        for level in levels:
            scores = heatmap_data.get((pillar, level), [])
            avg = sum(scores) / len(scores) if scores else 0
            row.append(round(avg, 1))
        matrix.append(row)
    
    # Create heatmap
    heatmap_df = pd.DataFrame(
        matrix,
        index=pillars,
        columns=[f"Level {i}" for i in levels]
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Level {i}" for i in levels],
        y=pillars,
        colorscale="RdYlGn",
        text=matrix,
        texttemplate="%{z:.1f}",
        textfont={"size": 14},
        zmin=0,
        zmax=10
    ))
    
    fig.update_layout(
        title="LLM Judge Score Heatmap (Pillar × Level)",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Color Legend:**
    - 🟢 **Green (8-10)**: Excellent - Model handles this well
    - 🟡 **Yellow (5-7)**: Moderate - Room for improvement  
    - 🔴 **Red (0-4)**: Poor - Significant issues need attention
    """)
    
    # Human vs Judge Heatmap
    st.markdown("### 👥 Human vs Judge Heatmap")
    
    human_matrix = []
    
    for pillar in pillars:
        row = []
        for level in levels:
            matches = [m for m in matched if m["pillar"] == pillar and m["level"] == level]
            if matches:
                avg = sum(m["human_score"] for m in matches) / len(matches)
                row.append(round(avg, 1))
            else:
                row.append(None)
        human_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=human_matrix,
        x=[f"Level {i}" for i in levels],
        y=pillars,
        colorscale="RdYlGn",
        text=human_matrix,
        texttemplate="%{z:.1f}" if human_matrix else "",
        textfont={"size": 14},
        zmin=0,
        zmax=10
    ))
    
    fig.update_layout(
        title="Human Expert Score Heatmap (Pillar × Level)",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Comparison Table
    st.markdown("### 📋 Detailed Comparison Table")
    
    comparison_df = pd.DataFrame(matched)
    comparison_df.columns = ["Prompt ID", "Pillar", "Level", "Judge Score", "Human Score", "Judge Reasoning", "Human Reasoning", "Verdict"]
    
    st.dataframe(
        comparison_df[["Prompt ID", "Pillar", "Level", "Judge Score", "Human Score", "Verdict"]],
        use_container_width=True
    )
    
    # Generate Final Report
    st.markdown("---")
    st.markdown("## 📄 Generate Final Comprehensive Report")
    
    if st.button("📥 Generate & Save Full Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            report = generate_final_comparison_report(evaluations, matched, judge_results, delta, pillar_comparison)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                os.path.dirname(__file__),
                "data", "reports", f"final_report_{timestamp}.md"
            )
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            
            st.success(f"✅ Report saved to: {report_path}")
            
            # Display report
            st.markdown(report)


def generate_final_comparison_report(
    evaluations: list,
    matched: list,
    judge_results: dict,
    delta: dict,
    pillar_comparison: dict
) -> str:
    """Generate the final comprehensive comparison report."""
    
    report = f"""
# Golden 68 - Final Comprehensive Evaluation Report

**Test Model:** {st.session_state.test_display_name} ({st.session_state.test_model_name})  
**Judge Model:** {st.session_state.judge_display_name} ({st.session_state.judge_model_name})  
**Human Auditor:** Expert  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Executive Summary

| Metric | LLM Judge | Human Expert |
|--------|-----------|--------------|
| Total Prompts | {len(evaluations)} | {len(matched)} |
| Average Score | {judge_results['overall_score']:.2f}/10 | {sum(m['human_score'] for m in matched)/len(matched) if matched else 0:.2f}/10 |
| Pass Rate | {judge_results['pass_rate']*100:.1f}% | {sum(1 for m in matched if m['human_score'] >= 6)/len(matched)*100 if matched else 0:.1f}% |

### 🎯 Agreement Analysis

| Metric | Value |
|--------|-------|
| **Agreement Delta** | {delta['agreement_delta']:.3f} |
| **Rating** | {delta['rating']} |
| **Mean Absolute Difference** | {delta['mean_absolute_difference']:.2f} |
| **Exact Agreement Rate** | {delta['exact_agreement_rate']*100:.1f}% |

---

## 📋 Pillar-by-Pillar Comparison

| Pillar | Judge Avg | Human Avg | Delta | Agreement |
|--------|-----------|-----------|-------|-----------|
"""
    
    for pillar, data in pillar_comparison.items():
        report += f"| {pillar} | {data['judge_avg']:.2f} | {data['human_avg']:.2f} | {data['delta']['agreement_delta']:.3f} | {data['delta']['rating']} |\n"
    
    report += f"""

---

## 🗺️ Regulatory Readiness Heatmap

The following heatmap shows model performance across pillars and complexity levels:

```
                  Level 1   Level 2   Level 3   Level 4   Level 5
Causality         [Score]   [Score]   [Score]   [Score]   [Score]
Compliance        [Score]   [Score]   [Score]   [Score]   [Score]
Consistency       [Score]   [Score]   [Score]   [Score]   [Score]
```

### Key Findings

"""
    
    # Find weakest areas
    heatmap_data = {}
    for eval_item in evaluations:
        key = (eval_item.get("pillar"), eval_item.get("level"))
        if key not in heatmap_data:
            heatmap_data[key] = []
        heatmap_data[key].append(eval_item.get("judge_score", 0))
    
    weakest = None
    weakest_score = 10
    for (pillar, level), scores in heatmap_data.items():
        avg = sum(scores) / len(scores)
        if avg < weakest_score:
            weakest_score = avg
            weakest = (pillar, level)
    
    strongest = None
    strongest_score = 0
    for (pillar, level), scores in heatmap_data.items():
        avg = sum(scores) / len(scores)
        if avg > strongest_score:
            strongest_score = avg
            strongest = (pillar, level)
    
    if weakest:
        report += f"""
- **Weakest Area:** {weakest[0]} at Level {weakest[1]} (Avg: {weakest_score:.1f}/10)
  - This combination of complexity and pillar category shows the most significant gaps.
"""
    
    if strongest:
        report += f"""
- **Strongest Area:** {strongest[0]} at Level {strongest[1]} (Avg: {strongest_score:.1f}/10)
  - The model demonstrates excellent capability in this area.
"""
    
    report += """

---

## 🔍 Detailed Issue Analysis

### Critical Failures Requiring Attention

"""
    
    failures = [e for e in evaluations if e["judge_determination"] == "FAIL"]
    
    for failure in failures[:5]:
        report += f"""
#### {failure['prompt_id']} - Score: {failure['judge_score']}/10

**Pillar:** {failure['pillar']} | **Level:** {failure['level']}

**Issue:** {failure['judge_reasoning'][:300]}

"""
    
    report += """

---

## 💡 Recommendations

### For Model Improvement:

"""
    
    # Analyze patterns
    pillar_scores = judge_results["pillar_scores"]
    weakest_pillar = min(pillar_scores.keys(), key=lambda p: pillar_scores[p])
    
    report += f"""
1. **Prioritize {weakest_pillar} Training**
   - This pillar showed the lowest average score ({pillar_scores[weakest_pillar]:.1f}/10)
   - Focus on improving {weakest_pillar.lower()}-related reasoning capabilities.

"""
    
    if delta["agreement_delta"] < 0.6:
        report += """
2. **Review LLM-as-Judge Calibration**
   - The judge shows significant disagreement with human experts.
   - Consider using a different judge model or adjusting the judge prompt.

"""
    
    level_scores = judge_results["level_scores"]
    if 5 in level_scores and level_scores[5]["average_score"] < 5:
        report += f"""
3. **Strengthen Adversarial Robustness**
   - Level 5 (adversarial) prompts achieved only {level_scores[5]['average_score']:.1f}/10
   - Implement additional safety measures and robustness training.

"""
    
    report += f"""

---

## 📁 Audit Trail

- **Total Evaluations Conducted:** {len(evaluations)}
- **Human Audits Completed:** {len(matched)}
- **Agreement Rate:** {delta['exact_agreement_rate']*100:.1f}%
- **Report Generated:** {datetime.now().isoformat()}

---

*This report was generated by Golden 68 - AI Compliance & Audit Framework*
"""
    
    return report


# ============================================
# HELPER FUNCTIONS
# ============================================
def save_evaluation_logs(logs: list, prefix: str = "evaluation"):
    """Save evaluation logs to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    
    log_dir = os.path.join(os.path.dirname(__file__), "data", "results")
    os.makedirs(log_dir, exist_ok=True)
    
    filepath = os.path.join(log_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "test_model": st.session_state.get("test_model_name", "unknown"),
            "judge_model": st.session_state.get("judge_model_name", "unknown"),
            "total": len(logs),
            "evaluations": logs
        }, f, indent=2, ensure_ascii=False)
    
    return filepath


# ============================================
# MAIN APP
# ============================================
def render_history_page():
    """Display history of all previous evaluations with model grouping and leaderboard."""
    st.markdown('<h1 class="main-header">📜 Evaluation History</h1>', unsafe_allow_html=True)
    
    results_dir = "data/results"
    
    # Get all result files
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    result_files.sort(reverse=True)
    
    if not result_files:
        st.info("📭 No evaluation history found. Run an evaluation first!")
        if st.button("← Back to Setup", type="primary"):
            st.session_state.current_step = "setup"
            st.rerun()
        return
    
    # Load all data and group by model
    all_data = {}
    model_stats = {}  # For leaderboard
    
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get model name from filename or data
        model_name = data.get('test_model', data.get('model', filename.replace('.json', '').split('_')[-1]))
        
        if model_name not in all_data:
            all_data[model_name] = []
        all_data[model_name].append({
            'filename': filename,
            'data': data,
            'avg_score': data.get('avg_score', 0),
            'completed': data.get('completed', 0)
        })
        
        # Update model stats for leaderboard
        results = data.get('results', [])
        if results:
            scores = [r.get('judge_score', 0) for r in results]
            passes = sum(1 for r in results if r.get('judge_determination') == 'PASS')
            avg = sum(scores) / len(scores) if scores else 0
            pass_rate = passes / len(results) * 100 if results else 0
            
            if model_name not in model_stats:
                model_stats[model_name] = {'scores': [], 'passes': 0, 'evaluations': 0}
            model_stats[model_name]['scores'].extend(scores)
            model_stats[model_name]['passes'] += passes
            model_stats[model_name]['evaluations'] += len(results)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📋 By Model", "🏆 Leaderboard"])
    
    with tab1:
        # Model selection
        model_names = list(all_data.keys())
        selected_model = st.selectbox("🤖 Select Model:", model_names, index=0)
        
        if selected_model:
            model_evaluations = all_data[selected_model]
            
            # Model summary
            stats = model_stats.get(selected_model, {})
            total_scores = stats.get('scores', [])
            total_evals = stats.get('evaluations', 0)
            total_passes = stats.get('passes', 0)
            overall_avg = sum(total_scores) / len(total_scores) if total_scores else 0
            overall_pass_rate = total_passes / total_evals * 100 if total_evals else 0
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Evaluations", len(model_evaluations))
            with col2:
                st.metric("Avg Score", f"{overall_avg:.2f}/10")
            with col3:
                st.metric("Pass Rate", f"{overall_pass_rate:.1f}%")
            with col4:
                grade = "A+" if overall_avg >= 9 else "A" if overall_avg >= 8 else "B" if overall_avg >= 7 else "C" if overall_avg >= 6 else "D"
                st.metric("Grade", grade)
            
            st.markdown("---")
            
            # Select specific evaluation
            eval_options = [f"{e['filename']} (Score: {e['avg_score']:.1f})" for e in model_evaluations]
            selected_eval = st.selectbox("📁 Select Evaluation:", eval_options, index=0)
            
            # Find selected evaluation data
            selected_data = next((e['data'] for e in model_evaluations if f"{e['filename']} (Score: {e['avg_score']:.1f})" == selected_eval), None)
            
            if selected_data:
                results = selected_data.get('results', [])
                completed = selected_data.get('completed', len(results))
                total = selected_data.get('total', len(results))
                avg_score = selected_data.get('avg_score', 0)
                passes = selected_data.get('passes', 0)
                
                # Sub-tabs for evaluation details
                sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📊 Overview", "📋 Results", "📈 Charts", "📝 Report"])
                
                with sub_tab1:
                    pillar_stats = {}
                    for r in results:
                        p = r.get('pillar', 'Unknown')
                        if p not in pillar_stats:
                            pillar_stats[p] = {'scores': [], 'passes': 0}
                        pillar_stats[p]['scores'].append(r.get('judge_score', 0))
                        if r.get('judge_determination') == 'PASS':
                            pillar_stats[p]['passes'] += 1
                    
                    for pillar, pstats in pillar_stats.items():
                        avg_p = sum(pstats['scores'])/len(pstats['scores']) if pstats['scores'] else 0
                        pr = pstats['passes']/len(pstats['scores'])*100 if pstats['scores'] else 0
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"🔗 {pillar}", f"{avg_p:.2f}/10")
                        with col2:
                            st.metric("Pass Rate", f"{pr:.1f}%")
                        with col3:
                            st.metric("Count", len(pstats['scores']))
                
                with sub_tab2:
                    df_data = []
                    for r in results:
                        df_data.append({
                            "Prompt": r.get('prompt_id', ''),
                            "Pillar": r.get('pillar', ''),
                            "Level": r.get('level', 0),
                            "Score": r.get('judge_score', 0),
                            "Status": "✅ PASS" if r.get('judge_determination') == 'PASS' else "❌ FAIL"
                        })
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, height=400)
                
                with sub_tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(pd.DataFrame(results), x="judge_score", nbins=10, 
                                          title="Score Distribution", color_discrete_sequence=["#667eea"])
                        fig.update_layout(xaxis_title="Score", yaxis_title="Count", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        pass_count = sum(1 for r in results if r.get('judge_determination') == 'PASS')
                        fail_count = len(results) - pass_count
                        fig = px.pie(values=[pass_count, fail_count], names=["PASS", "FAIL"],
                                    title="Pass/Fail", color=["#28a745", "#dc3545"], hole=0.4)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with sub_tab4:
                    report_md = f"""# Golden 68 Evaluation Report

## {selected_model}
- **Evaluation**: {selected_eval.split(' (')[0]}
- **Total Prompts**: {completed}/{total}
- **Average Score**: {avg_score:.2f}/10
- **Pass Rate**: {passes}/{completed} ({passes/completed*100:.1f}%)

"""
                    st.markdown(report_md)
                    st.download_button("📥 Download Report", report_md, 
                                     file_name=f"report_{selected_model.replace(' ', '_')}.md", mime="text/markdown")
    
    with tab2:
        # Leaderboard
        st.markdown("### 🏆 Model Leaderboard")
        
        # Build leaderboard data
        leaderboard = []
        for model, stats in model_stats.items():
            scores = stats.get('scores', [])
            passes = stats.get('passes', 0)
            evals = stats.get('evaluations', 0)
            avg_score = sum(scores) / len(scores) if scores else 0
            pass_rate = passes / evals * 100 if evals else 0
            
            # Calculate weighted score (70% avg score + 30% pass rate)
            weighted = (avg_score / 10 * 70) + (pass_rate * 0.30)
            
            leaderboard.append({
                'Model': model,
                'Avg Score': round(avg_score, 2),
                'Pass Rate': round(pass_rate, 1),
                'Total Prompts': evals,
                'Weighted Score': round(weighted, 2)
            })
        
        # Sort by weighted score
        leaderboard.sort(key=lambda x: x['Weighted Score'], reverse=True)
        
        # Add rank
        for i, entry in enumerate(leaderboard):
            entry['Rank'] = i + 1
        
        lb_df = pd.DataFrame(leaderboard)
        
        # Display leaderboard with rank badges
        st.markdown("#### Model Rankings (Weighted Score: 70% Avg Score + 30% Pass Rate)")
        
        for _, row in lb_df.iterrows():
            rank_emoji = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉" if row['Rank'] == 3 else f"#{row['Rank']}"
            grade = "A+" if row['Avg Score'] >= 9 else "A" if row['Avg Score'] >= 8 else "B" if row['Avg Score'] >= 7 else "C" if row['Avg Score'] >= 6 else "D"
            
            with st.expander(f"{rank_emoji} {row['Model']} - Score: {row['Avg Score']}/10 (Grade: {grade})"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Score", f"{row['Avg Score']}/10")
                with col2:
                    st.metric("Pass Rate", f"{row['Pass Rate']}%")
                with col3:
                    st.metric("Total Prompts", row['Total Prompts'])
                with col4:
                    st.metric("Weighted Score", f"{row['Weighted Score']}")
        
        # Leaderboard chart
        st.markdown("#### 📊 Comparison Chart")
        fig = px.bar(lb_df, x="Model", y="Avg Score", color="Rank", 
                    title="Model Performance Comparison", 
                    color_continuous_scale="RdYlGn")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export leaderboard
        export_md = "# Golden 68 Model Leaderboard\n\n"
        export_md += "| Rank | Model | Avg Score | Pass Rate | Total Prompts | Weighted Score |\n"
        export_md += "|------|-------|-----------|-----------|---------------|----------------|\n"
        for row in leaderboard:
            export_md += f"| {row['Rank']} | {row['Model']} | {row['Avg Score']}/10 | {row['Pass Rate']}% | {row['Total Prompts']} | {row['Weighted Score']} |\n"
        
        st.download_button("📥 Export Leaderboard", export_md, 
                           file_name="model_leaderboard.md", mime="text/markdown")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Setup", type="secondary"):
            st.session_state.current_step = "setup"
            st.rerun()
    with col2:
        if st.button("🔄 Run New Evaluation →", type="primary"):
            st.session_state.current_step = "setup"
            st.rerun()


# ============================================
# COST MONITOR PAGE
# ============================================
def render_cost_monitor_page():
    """Cost Monitor Dashboard - Track API usage and credits."""
    st.markdown('<p class="main-header">💰 Cost Monitor Dashboard</p>', unsafe_allow_html=True)
    
    # Initialize cost tracker in session state
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = APICostTracker()
    
    cost_tracker = st.session_state.cost_tracker
    
    # Initialize API keys from session state
    api_keys = {}
    if "judge_api_key" in st.session_state and st.session_state.judge_api_key:
        api_keys["gemini"] = st.session_state.judge_api_key
    if "test_api_key" in st.session_state and st.session_state.test_api_key:
        provider = st.session_state.get("test_provider", "openrouter")
        api_keys[provider] = st.session_state.test_api_key
    if "nvidia_api_key" in st.session_state and st.session_state.nvidia_api_key:
        api_keys["nvidia"] = st.session_state.nvidia_api_key
    
    # Refresh usage stats
    for provider, key in api_keys.items():
        cost_tracker.track_request(provider, key, tokens_used=0)
    
    # Overview metrics
    st.markdown("### 📊 Usage Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_keys = len(cost_tracker.usage_history)
    total_requests = sum(len(v) for v in cost_tracker.usage_history.values())
    total_tokens = sum(sum(u.get("tokens_used", 0) for u in v) for v in cost_tracker.usage_history.values())
    
    with col1:
        st.metric("Active API Keys", total_keys)
    with col2:
        st.metric("Total Requests", total_requests)
    with col3:
        st.metric("Total Tokens", f"{total_tokens:,}")
    with col4:
        estimated_cost = total_tokens * 0.00001  # Rough estimate
        st.metric("Est. Cost", f"${estimated_cost:.4f}")
    
    st.markdown("---")
    
    # API Keys Details
    st.markdown("### 🔑 API Key Usage Details")
    
    # Group by provider
    provider_summary = {}
    for provider, key in api_keys.items():
        key_prefix = key[:8] + "..." if len(key) > 8 else key
        requests = len(cost_tracker.usage_history.get(provider, []))
        tokens = sum(u.get("tokens_used", 0) for u in cost_tracker.usage_history.get(provider, []))
        limits = cost_tracker.PROVIDER_LIMITS.get(provider, {})
        
        provider_summary[provider] = {
            "key_preview": key_prefix,
            "requests": requests,
            "tokens": tokens,
            "rpm": limits.get("rpm", "N/A"),
            "tpm": limits.get("tpm", "N/A"),
            "daily_limit": limits.get("daily_limit", "N/A"),
            "cost_per_1k": limits.get("cost_per_1k", 0)
        }
    
    if provider_summary:
        summary_df = pd.DataFrame([
            {"Provider": p, "Key Preview": v["key_preview"], "Requests": v["requests"], 
             "Tokens Used": v["tokens"], "RPM Limit": v["rpm"], "TPM Limit": v["tpm"],
             "Daily Limit": v["daily_limit"], "Cost/1K Tokens": f"${v['cost_per_1k']:.4f}"}
            for p, v in provider_summary.items()
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No API keys configured yet. Configure API keys in the Setup page to track usage.")
    
    st.markdown("---")
    
    # Rate Limit Status
    st.markdown("### 🚦 Rate Limit Status")
    
    for provider, info in provider_summary.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            rpm = info["rpm"]
            tpm = info["tpm"]
            daily = info["daily_limit"]
            
            if isinstance(rpm, int) and info["requests"] > 0:
                rpm_pct = min(100, (info["requests"] / (rpm * 60)) * 100)  # Rough estimate
            else:
                rpm_pct = 0
            
            st.markdown(f"**{provider.upper()}**")
            st.progress(min(1.0, rpm_pct / 100), text=f"Rate: {info['requests']} requests (Est. {rpm_pct:.1f}% of limit)")
        
        with col2:
            st.markdown(f"**Limits:**")
            st.caption(f"RPM: {rpm}")
            st.caption(f"TPM: {tpm}")
            st.caption(f"Daily: {daily:,}" if isinstance(daily, int) else f"Daily: {daily}")
    
    st.markdown("---")
    
    # Usage History Chart
    st.markdown("### 📈 Usage History")
    
    if cost_tracker.usage_history:
        # Flatten history for chart
        history_data = []
        for provider, records in cost_tracker.usage_history.items():
            for record in records:
                history_data.append({
                    "Provider": provider,
                    "Timestamp": record.get("timestamp", datetime.now()),
                    "Tokens": record.get("tokens_used", 0),
                    "Success": record.get("success", True)
                })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            
            tab1, tab2 = st.tabs(["Tokens Over Time", "Requests by Provider"])
            with tab1:
                fig = px.line(history_df, x="Timestamp", y="Tokens", color="Provider",
                             title="Token Usage Over Time", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                provider_counts = history_df.groupby("Provider").size().reset_index(name="Requests")
                fig = px.pie(provider_counts, values="Requests", names="Provider",
                            title="Requests Distribution")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usage history yet. Run evaluations to see usage statistics.")
    
    st.markdown("---")
    
    # Cost Estimations
    st.markdown("### 💵 Cost Estimations")
    
    cost_data = []
    for provider, info in provider_summary.items():
        tokens = info["tokens"]
        cost_per_1k = info["cost_per_1k"]
        estimated = (tokens / 1000) * cost_per_1k
        cost_data.append({
            "Provider": provider.upper(),
            "Tokens Used": tokens,
            "Cost per 1K": f"${cost_per_1k:.4f}",
            "Estimated Cost": f"${estimated:.4f}"
        })
    
    if cost_data:
        cost_df = pd.DataFrame(cost_data)
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
        
        total_estimated = sum((d["tokens"] / 1000) * provider_summary[p.lower()]["cost_per_1k"] 
                            for p, d in zip(cost_df["Provider"], cost_data) 
                            if p.lower() in provider_summary)
        st.markdown(f"**Total Estimated Cost: ${total_estimated:.4f}**")
    else:
        st.info("No cost data available.")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Setup", type="secondary", use_container_width=True):
            st.session_state.current_step = "setup"
            st.rerun()
    with col2:
        if st.button("📜 View History", type="secondary", use_container_width=True):
            st.session_state.current_step = "history"
            st.rerun()
    with col3:
        if st.button("📊 Model Comparison", type="secondary", use_container_width=True):
            st.session_state.current_step = "comparison"
            st.rerun()


# ============================================
def main():
    """Main application entry point."""
    init_session_state()
    
    # Step-based navigation
    if st.session_state.current_step == "setup":
        render_setup_page()
    elif st.session_state.current_step == "evaluation":
        render_evaluation_page()
    elif st.session_state.current_step == "human_audit":
        render_human_audit_page()
    elif st.session_state.current_step == "comparison":
        render_comparison_page()
    elif st.session_state.current_step == "history":
        render_history_page()
    elif st.session_state.current_step == "cost_monitor":
        render_cost_monitor_page()
    
    # Debug: Reset button
    with st.expander("🔧 Debug Options"):
        if st.button("🔄 Reset All Progress"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
