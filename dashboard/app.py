import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import shutil
import subprocess
import sys

st.set_page_config(layout="wide", page_title="Retention Decision Engine")

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
DATASET_DIR = ROOT_DIR / "Datasets"
REQUIRED_OUTPUTS = [
    OUTPUT_DIR / "master_action_table.csv",
    OUTPUT_DIR / "outreach_outcomes.csv",
]


def _outputs_ready():
    return all(path.exists() for path in REQUIRED_OUTPUTS)


def _prepare_directories():
    (ROOT_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _seed_raw_data_if_missing():
    if not DATASET_DIR.exists():
        return
    for csv_path in DATASET_DIR.glob("*.csv"):
        target = RAW_DATA_DIR / csv_path.name
        if not target.exists():
            shutil.copy2(csv_path, target)


def _run_module(module_name):
    cmd = [sys.executable, "-m", module_name]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )


def bootstrap_outputs_if_missing():
    if _outputs_ready():
        return

    _prepare_directories()
    _seed_raw_data_if_missing()
    _run_module("src.models.churn_model")
    _run_module("src.models.channel_model")
    _run_module("src.pipelines.full_pipeline")

    if not _outputs_ready():
        raise RuntimeError("Pipeline finished but required output files are still missing.")

@st.cache_data
def load_master_data():
    try:
        df = pd.read_csv(OUTPUT_DIR / "master_action_table.csv")
        outcomes = pd.read_csv(OUTPUT_DIR / "outreach_outcomes.csv")
        return df, outcomes
    except FileNotFoundError:
        st.error("Pipeline outputs are missing. Use the generate button below.")
        return pd.DataFrame(), pd.DataFrame()

st.title("🛡️ Predictive Customer Outreach and Retention Engine")
st.markdown("---")

if not _outputs_ready():
    st.warning("Required output files are not available yet.")
    if st.button("Generate Outputs Now (may take several minutes)"):
        with st.spinner("Running models and pipeline..."):
            try:
                bootstrap_outputs_if_missing()
                st.cache_data.clear()
                st.success("Pipeline outputs generated. Reloading dashboard...")
                st.rerun()
            except Exception as ex:
                st.error("Failed to generate pipeline outputs.")
                st.exception(ex)

df_master, df_outcomes = load_master_data()

if not df_master.empty:
    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers Evaluated", len(df_master))
    
    critical_count = len(df_master[df_master['risk_band'] == 'critical_risk'])
    col2.metric("Critical Risk Customers", critical_count)
    
    high_count = len(df_master[df_master['risk_band'] == 'high_risk'])
    col3.metric("High Risk Customers", high_count)
    
    outreach_perc = len(df_outcomes[df_outcomes['outreach_sent'] == True]) / len(df_outcomes) * 100
    col4.metric("Actionable Outreach %", f"{outreach_perc:.1f}%")

    st.markdown("---")
    
    # Layout with 2 columns
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Customer Risk Segmentation")
        # Donut chart of Risk Bands
        fig_risk = px.pie(df_master, names='risk_band', hole=0.4, 
                          color='risk_band',
                          color_discrete_map={
                              'critical_risk': '#ff4b4b',
                              'high_risk': '#ffa64b',
                              'medium_risk': '#ffd14b',
                              'low_risk': '#4caf50'
                          })
        fig_risk.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_risk, width='stretch')
        
    with right_col:
        st.subheader("Top Driven Reasons for Churn (SHAP)")
        # Bar chart of top_reason_1
        reason_counts = df_master['top_reason_1'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        fig_reasons = px.bar(reason_counts.head(7), x='Count', y='Reason', orientation='h',
                             color='Count', color_continuous_scale='Reds')
        fig_reasons.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_reasons, width='stretch')

    st.markdown("---")
    
    # Action Recommendation Table
    st.subheader("Targeted Outreach & Action Plan")
    filter_band = st.selectbox("Filter by Risk Band", ['All', 'critical_risk', 'high_risk', 'medium_risk', 'low_risk'])
    
    display_df = df_master.copy()
    if filter_band != 'All':
        display_df = display_df[display_df['risk_band'] == filter_band]
        
    display_cols = ['customer_id', 'churn_score', 'risk_band', 'segment_name', 
                    'top_reason_1', 'execution_channel', 'chosen_action', 'action_note']
    
    st.dataframe(display_df[display_cols].sort_values(by='churn_score', ascending=False).head(50), 
                 width='stretch')
                 
    st.markdown("---")
    
    # Outcome Simulation / Feedback Loop display
    st.subheader("Outcome Feedback Loop (Simulated)")
    st.markdown("Monitoring response rates to actions taken.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        # Retention rate by action
        agg_outcomes = df_outcomes[df_outcomes['outreach_sent']==True].groupby('chosen_action')['retained_90d'].mean().reset_index()
        agg_outcomes['retained_90d'] = agg_outcomes['retained_90d'] * 100
        fig_out = px.bar(agg_outcomes, x='chosen_action', y='retained_90d', title='Save Rate by Action (%)', text='retained_90d')
        fig_out.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_out, width='stretch')
        
    with sim_col2:
        # Retention rate by channel
        agg_chan = df_outcomes[df_outcomes['outreach_sent']==True].groupby('execution_channel')['retained_90d'].mean().reset_index()
        agg_chan['retained_90d'] = agg_chan['retained_90d'] * 100
        fig_chan = px.bar(agg_chan, x='execution_channel', y='retained_90d', title='Save Rate by Channel (%)', text='retained_90d')
        fig_chan.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_chan, width='stretch')
