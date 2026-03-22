import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from src.models.channel_model import ChannelModel
from src.models.churn_model import ChurnModel
from src.rules.action_engine import ActionEngine


st.set_page_config(
    layout="wide",
    page_title="Retention Decision Engine",
    page_icon="\U0001F6E1\uFE0F",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
DATASET_DIR = ROOT_DIR / "Datasets"
REQUIRED_OUTPUTS = [
    OUTPUT_DIR / "master_action_table.csv",
    OUTPUT_DIR / "outreach_outcomes.csv",
]

RISK_ORDER = ["critical_risk", "high_risk", "medium_risk", "low_risk"]
RISK_COLORS = {
    "critical_risk": "#D62839",
    "high_risk": "#F77F00",
    "medium_risk": "#FCBF49",
    "low_risk": "#2A9D8F",
}
CHANNEL_CHOICES = ["Email", "SMS", "Call", "Push", "In-App"]


def _channel_to_ui(value):
    if value is None:
        return "Email"
    raw = str(value).strip().lower()
    mapping = {
        "email": "Email",
        "sms": "SMS",
        "call": "Call",
        "phone": "Call",
        "phone_call": "Call",
        "push": "Push",
        "push_notification": "Push",
        "in-app": "In-App",
        "in_app": "In-App",
    }
    return mapping.get(raw, "Email")


def _channel_from_ui(value):
    raw = str(value).strip().lower()
    mapping = {
        "email": "email",
        "sms": "SMS",
        "call": "phone_call",
        "push": "push_notification",
        "in-app": "email",
    }
    return mapping.get(raw, "email")


def apply_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

            :root {
                --bg-1: #f4f1ea;
                --bg-2: #ffffff;
                --ink: #1f2933;
                --muted: #52606d;
                --accent: #0b6e4f;
                --accent-soft: #e4f2ee;
                --danger: #d62839;
                --warning: #f77f00;
                --radius-xl: 22px;
                --radius-md: 14px;
                --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.08);
            }

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
                color: var(--ink);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 8% 4%, #fcefd6 0%, rgba(252, 239, 214, 0) 32%),
                    radial-gradient(circle at 95% 22%, #dcebe4 0%, rgba(220, 235, 228, 0) 28%),
                    linear-gradient(180deg, var(--bg-1), #f7f7f5 45%, #fdfdfd);
            }

            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0.65);
                backdrop-filter: blur(8px);
            }

            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.02em;
            }

            .hero-wrap {
                background: linear-gradient(130deg, #102a43 0%, #0b6e4f 55%, #14532d 100%);
                color: #f8fafc;
                border-radius: var(--radius-xl);
                padding: 1.75rem 1.5rem;
                box-shadow: var(--shadow-soft);
                margin-bottom: 1.1rem;
            }

            .hero-kicker {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                opacity: 0.85;
                margin-bottom: 0.2rem;
            }

            .hero-title {
                margin: 0;
                font-size: 2rem;
                line-height: 1.15;
            }

            .hero-sub {
                margin-top: 0.7rem;
                margin-bottom: 0;
                opacity: 0.92;
                max-width: 900px;
                font-size: 1rem;
            }

            .section-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: var(--radius-xl);
                padding: 1rem 1rem 0.6rem 1rem;
                box-shadow: var(--shadow-soft);
                margin-bottom: 1rem;
            }

            .kpi-note {
                border-left: 4px solid var(--accent);
                background: var(--accent-soft);
                border-radius: 10px;
                padding: 0.65rem 0.9rem;
                font-size: 0.92rem;
                color: #1f2933;
            }

            .status-chip {
                display: inline-block;
                font-size: 0.75rem;
                border-radius: 999px;
                padding: 0.25rem 0.6rem;
                margin-right: 0.4rem;
                margin-top: 0.25rem;
                background: #edf2f7;
                color: #334e68;
            }

            .chip-danger { background: #fde8ea; color: #9f1239; }
            .chip-warning { background: #fff4e6; color: #9a3412; }
            .chip-success { background: #e7f8f2; color: #065f46; }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 12px;
                border: 1px solid #0b6e4f;
                background: #0b6e4f;
                color: #ffffff;
                font-weight: 700;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background: #09563e;
                border-color: #09563e;
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: var(--radius-md);
                padding: 0.7rem;
            }

            @media (max-width: 1024px) {
                .hero-title {
                    font-size: 1.6rem;
                }
                .hero-sub {
                    font-size: 0.95rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
            target.write_bytes(csv_path.read_bytes())


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


def bootstrap_outputs_with_status():
    if _outputs_ready():
        return

    _prepare_directories()
    _seed_raw_data_if_missing()

    status = st.status("Generating model outputs", expanded=True)
    with status:
        st.write("Preparing directories and seed datasets...")
        time.sleep(0.3)
        st.write("Training and exporting churn model assets...")
        _run_module("src.models.churn_model")
        st.write("Training and exporting channel model assets...")
        _run_module("src.models.channel_model")
        st.write("Running full retention pipeline...")
        _run_module("src.pipelines.full_pipeline")

    if not _outputs_ready():
        status.update(label="Output generation failed", state="error", expanded=True)
        raise RuntimeError("Pipeline finished but required output files are still missing.")
    status.update(label="Outputs generated successfully", state="complete", expanded=False)


def ensure_columns(df, defaults):
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
    return df


def _load_model_config():
    config_path = ROOT_DIR / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _artifact_paths_from_config(config):
    artifact_folder = ROOT_DIR / config["paths"]["artifacts"]["folder"]
    churn_model_path = artifact_folder / "churn_model.pkl"
    churn_encoder_path = artifact_folder / "churn_encoders.pkl"
    channel_model_path = artifact_folder / "channel_model.pkl"
    channel_encoder_path = artifact_folder / "channel_encoders.pkl"
    return {
        "churn_model": churn_model_path,
        "churn_encoder": churn_encoder_path,
        "channel_model": channel_model_path,
        "channel_encoder": channel_encoder_path,
    }


def _artifacts_ready(config):
    paths = _artifact_paths_from_config(config)
    return all(path.exists() for path in paths.values())


def _derive_reason_labels(df):
    score = pd.to_numeric(df.get("churn_score", 0.4), errors="coerce").fillna(0.4)
    balance = pd.to_numeric(df.get("Balance", df.get("balance", 0.0)), errors="coerce").fillna(0.0)
    inactive = pd.to_numeric(
        df.get("Months_Inactive_12_mon", df.get("Inactive_days", 0.0)),
        errors="coerce",
    ).fillna(0.0)

    reason = np.where(score >= 0.78, "declining_balance", "falling_spend_trend")
    reason = np.where(balance < balance.quantile(0.2), "credit_stress_signal", reason)
    reason = np.where(inactive > inactive.quantile(0.7), "low_account_activity", reason)
    reason = np.where((score >= 0.55) & (inactive > inactive.quantile(0.5)), "transaction_frequency_drop", reason)
    return pd.Series(reason, index=df.index)


def run_model_inference(df_input, selected_channels):
    cfg = _load_model_config()
    if not _artifacts_ready(cfg):
        return None, None, "Model artifacts missing. Using simulated scoring mode."

    churn_model = ChurnModel()
    channel_model = ChannelModel()
    artifacts = _artifact_paths_from_config(cfg)

    churn_model.load(str(artifacts["churn_model"]), str(artifacts["churn_encoder"]))
    channel_model.load(str(artifacts["channel_model"]), str(artifacts["channel_encoder"]))

    work = df_input.copy()
    if work.empty:
        return None, None, "Input is empty. Using simulated scoring mode."

    id_candidates = ["customer_id", "CLIENTNUM", "CustomerId", "customerId", "id"]
    id_col = next((c for c in id_candidates if c in work.columns), None)
    if id_col is None:
        work["customer_id"] = np.arange(100000, 100000 + len(work))
    else:
        work["customer_id"] = work[id_col]

    churn_scores = churn_model.predict_proba(work)
    score_df = pd.DataFrame(
        {
            "customer_id": work["customer_id"],
            "churn_score": churn_scores,
        }
    )
    score_df = churn_model.attach_risk_bands(score_df, cfg["churn_model"]["thresholds"])

    channel_input = work.copy()
    channel_input["churn_score"] = churn_scores
    if "segment_name" not in channel_input.columns:
        bal = pd.to_numeric(
            channel_input.get("Total_Revolving_Bal", channel_input.get("Balance", 0)),
            errors="coerce",
        ).fillna(0)
        channel_input["segment_name_encoded"] = np.where(
            bal > 100000,
            "premium_active",
            np.where(bal < 20000, "dormant_low_value", "at_risk_mid_tier"),
        )
    else:
        channel_input["segment_name_encoded"] = channel_input["segment_name"]

    ch_pred, ch_conf = channel_model.predict(channel_input)
    model_channel = pd.Series(ch_pred).map(_channel_to_ui)
    if selected_channels:
        allowed = set(selected_channels)
        fallback = selected_channels[0]
        model_channel = model_channel.map(lambda c: c if c in allowed else fallback)

    master = pd.DataFrame(
        {
            "customer_id": score_df["customer_id"],
            "churn_score": score_df["churn_score"],
            "risk_band": score_df["risk_band"].astype(str),
            "segment_name": channel_input["segment_name_encoded"].astype(str),
            "top_reason_1": _derive_reason_labels(channel_input),
            "best_channel": model_channel.map(_channel_from_ui),
            "channel_confidence": ch_conf,
        }
    )

    action_engine = ActionEngine(config_path=str(ROOT_DIR / "configs" / "rules_config.yaml"))
    action_values = master.apply(action_engine.determine_action, axis=1)
    action_values.columns = ["chosen_action", "action_priority", "action_note", "execution_channel"]
    master = pd.concat([master, action_values], axis=1)
    master["execution_channel"] = master["execution_channel"].map(_channel_to_ui)

    if selected_channels:
        allowed = set(selected_channels)
        fallback = selected_channels[0]
        master["execution_channel"] = master["execution_channel"].map(lambda c: c if c in allowed else fallback)

    rng = np.random.default_rng(11)
    outcomes = master[["customer_id", "risk_band", "execution_channel", "chosen_action", "top_reason_1"]].copy()
    outcomes.rename(columns={"top_reason_1": "predicted_reason"}, inplace=True)
    outcomes["timestamp"] = pd.Timestamp.today().normalize() - pd.to_timedelta(
        rng.integers(0, 28, size=len(outcomes)),
        unit="d",
    )
    outcomes["outreach_sent"] = outcomes["chosen_action"] != "MONITOR_ONLY"
    outcomes["responded"] = outcomes["outreach_sent"] & (rng.random(len(outcomes)) > 0.45)
    outcomes["retained_90d"] = outcomes["responded"] | (outcomes["risk_band"] == "low_risk")

    master = ensure_columns(
        master,
        {
            "execution_channel": "Email",
            "chosen_action": "EMAIL_NUDGE",
            "action_note": "General recommendation",
        },
    )
    return master, outcomes, "Live model inference enabled using trained artifacts."


@st.cache_data(show_spinner=False)
def load_output_data():
    if not _outputs_ready():
        return pd.DataFrame(), pd.DataFrame()

    master = pd.read_csv(OUTPUT_DIR / "master_action_table.csv")
    outcomes = pd.read_csv(OUTPUT_DIR / "outreach_outcomes.csv")
    return master, outcomes


@st.cache_data(show_spinner=False)
def load_demo_base_data():
    source = DATASET_DIR / "ChurnPrediction.csv"
    if source.exists():
        return pd.read_csv(source)

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "customer_id": np.arange(100001, 100251),
            "Balance": rng.normal(loc=68000, scale=21000, size=250).clip(min=2500),
            "Inactive_days": rng.integers(5, 120, size=250),
            "TenureMonths": rng.integers(3, 72, size=250),
        }
    )


def normalize_input_to_master(df_input, selected_channels):
    rng = np.random.default_rng(7)
    df = df_input.copy()

    id_candidates = ["customer_id", "CLIENTNUM", "CustomerId", "customerId", "id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        df["customer_id"] = np.arange(100000, 100000 + len(df))
    else:
        df["customer_id"] = df[id_col]

    balance_col = next((c for c in ["Total_Revolving_Bal", "Balance", "balance"] if c in df.columns), None)
    inactive_col = next((c for c in ["Months_Inactive_12_mon", "Inactive_days", "inactive_days"] if c in df.columns), None)

    if balance_col is None:
        bal = pd.Series(rng.normal(loc=65000, scale=20000, size=len(df)), index=df.index)
    else:
        bal = pd.to_numeric(df[balance_col], errors="coerce").fillna(df[balance_col].median())

    if inactive_col is None:
        inactivity = pd.Series(rng.integers(0, 120, size=len(df)), index=df.index)
    else:
        inactivity = pd.to_numeric(df[inactive_col], errors="coerce").fillna(0)

    bal_norm = (bal - bal.min()) / (bal.max() - bal.min() + 1e-9)
    inactive_norm = (inactivity - inactivity.min()) / (inactivity.max() - inactivity.min() + 1e-9)

    if "churn_score" in df.columns:
        churn_score = pd.to_numeric(df["churn_score"], errors="coerce").fillna(0.4).clip(0, 1)
    else:
        churn_score = (0.65 * inactive_norm + 0.35 * (1 - bal_norm)).clip(0, 1)

    risk_band = pd.cut(
        churn_score,
        bins=[-0.01, 0.35, 0.55, 0.75, 1.01],
        labels=["low_risk", "medium_risk", "high_risk", "critical_risk"],
    ).astype(str)

    segment_name = np.where(
        bal > 100000,
        "premium_active",
        np.where(bal < 20000, "dormant_low_value", "at_risk_mid_tier"),
    )

    reason = np.where(inactivity > inactivity.median(), "Declining Engagement", "Fee Sensitivity")
    reason = np.where(bal < bal.quantile(0.2), "Low Wallet Share", reason)

    channel_pool = selected_channels if selected_channels else CHANNEL_CHOICES
    exec_channel = rng.choice(channel_pool, size=len(df), replace=True)

    action = np.where(
        churn_score >= 0.75,
        "DISCOUNT_OFFER",
        np.where(churn_score >= 0.55, "CALL_RETENTION", np.where(churn_score >= 0.35, "ENGAGEMENT_NUDGE", "MONITOR_ONLY")),
    )
    action_note = np.where(
        action == "DISCOUNT_OFFER",
        "Offer premium retention bundle",
        np.where(action == "CALL_RETENTION", "Trigger advisor callback", np.where(action == "ENGAGEMENT_NUDGE", "Send personalized content", "Track behavior only")),
    )

    master = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "churn_score": churn_score,
            "risk_band": risk_band,
            "segment_name": segment_name,
            "top_reason_1": reason,
            "execution_channel": exec_channel,
            "chosen_action": action,
            "action_note": action_note,
        }
    )

    outcomes = master[["customer_id", "risk_band", "execution_channel", "chosen_action"]].copy()
    outcomes["timestamp"] = pd.Timestamp.today().normalize() - pd.to_timedelta(rng.integers(0, 28, size=len(outcomes)), unit="d")
    outcomes["outreach_sent"] = outcomes["chosen_action"] != "MONITOR_ONLY"
    outcomes["responded"] = outcomes["outreach_sent"] & (rng.random(len(outcomes)) > 0.45)
    outcomes["retained_90d"] = outcomes["responded"] | (outcomes["risk_band"] == "low_risk")

    return master, outcomes


def process_campaign(master_df, outcomes_df, campaign_cfg):
    work = master_df.copy()
    work = ensure_columns(
        work,
        {
            "churn_score": 0.4,
            "risk_band": "medium_risk",
            "segment_name": "at_risk_mid_tier",
            "top_reason_1": "Unknown",
            "execution_channel": "Email",
            "chosen_action": "ENGAGEMENT_NUDGE",
            "action_note": "Follow default policy",
        },
    )

    risk_filter = campaign_cfg["risk_filter"]
    channel_filter = campaign_cfg["channels"]
    max_customers = campaign_cfg["max_customers"]

    work = work[work["risk_band"].isin(risk_filter)]
    work = work[work["execution_channel"].isin(channel_filter)]

    risk_weight = {
        "critical_risk": 1.00,
        "high_risk": 0.8,
        "medium_risk": 0.55,
        "low_risk": 0.3,
    }
    work["priority_score"] = work["churn_score"].clip(0, 1) * work["risk_band"].map(risk_weight).fillna(0.5)
    work = work.sort_values("priority_score", ascending=False).head(max_customers)

    outcomes = outcomes_df.copy()
    outcomes = ensure_columns(
        outcomes,
        {
            "timestamp": pd.Timestamp.today().normalize(),
            "outreach_sent": True,
            "responded": False,
            "retained_90d": False,
        },
    )
    outcomes["timestamp"] = pd.to_datetime(outcomes["timestamp"], errors="coerce").fillna(pd.Timestamp.today().normalize())
    outcomes = outcomes[outcomes["customer_id"].isin(work["customer_id"])].copy()

    return work, outcomes


def render_header():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-kicker">Retention Intelligence Studio</div>
            <h1 class="hero-title">Predictive Customer Outreach and Retention Engine</h1>
            <p class="hero-sub">
                End-to-end decision workflow: intake customer data, process campaign rules, and review performance in a compact executive dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls(outputs_ready):
    with st.sidebar:
        st.markdown("### Workspace Controls")
        st.caption("Use pipeline outputs or run the full model pipeline directly from the dashboard.")

        if outputs_ready:
            st.success("Outputs are available and ready.")
        else:
            st.warning("Outputs missing. You can still use demo mode.")

        if st.button("Generate Pipeline Outputs", use_container_width=True):
            try:
                bootstrap_outputs_with_status()
                st.cache_data.clear()
                st.success("Outputs generated. Dashboard refreshed.")
                st.rerun()
            except Exception as ex:
                st.error("Pipeline generation failed.")
                st.exception(ex)

        st.divider()
        st.caption("Tip: Upload CSV or run with demo data for quick walkthroughs.")


def collect_user_input(output_master_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("1. Input Data")
    st.caption("Provide campaign settings and choose where customer data should come from.")

    source_mode = st.radio(
        "Choose input source",
        ["Use pipeline outputs", "Upload CSV", "Use demo sample"],
        horizontal=True,
    )

    input_df = pd.DataFrame()
    source_note = ""

    if source_mode == "Use pipeline outputs":
        if output_master_df.empty:
            st.info("Pipeline outputs are not available, so demo data will be used automatically.")
            input_df = load_demo_base_data()
            source_note = "Demo fallback used because pipeline outputs are missing"
        else:
            input_df = output_master_df.copy()
            source_note = "Using generated pipeline outputs"

    elif source_mode == "Upload CSV":
        upload = st.file_uploader("Upload customer CSV", type=["csv"])
        if upload is not None:
            try:
                input_df = pd.read_csv(upload)
                st.success(f"Uploaded {len(input_df):,} records.")
                st.dataframe(input_df.head(10), use_container_width=True)
                source_note = "Using user uploaded file"
            except Exception as ex:
                st.error("Unable to parse the uploaded CSV.")
                st.exception(ex)
        else:
            st.warning("No file uploaded yet. Use demo sample to preview the dashboard.")

    else:
        input_df = load_demo_base_data()
        source_note = "Using built-in demo data"
        st.info("Demo sample loaded so the dashboard always has meaningful visuals.")

    c1, c2 = st.columns(2)
    with c1:
        campaign_name = st.text_input("Campaign name", value="Q2 Save Plan")
        campaign_objective = st.selectbox(
            "Campaign objective",
            ["Reduce churn", "Increase engagement", "Boost high-value retention", "Win-back dormant users"],
        )
        run_date = st.date_input("Planned run date")
        budget = st.number_input("Incentive budget (USD)", min_value=0, value=25000, step=500)

    with c2:
        risk_filter = st.multiselect("Risk bands to target", options=RISK_ORDER, default=RISK_ORDER[:3])
        channels = st.multiselect("Channels to allow", options=CHANNEL_CHOICES, default=["Email", "SMS", "Call"])
        max_customers = st.slider("Max customers to include", min_value=25, max_value=1000, value=250, step=25)
        operator_note = st.text_area("Operator note", value="Prioritize critical-risk users with omnichannel outreach.")

    submit = st.button("Process Campaign", type="primary", use_container_width=True)

    config = {
        "campaign_name": campaign_name,
        "campaign_objective": campaign_objective,
        "run_date": run_date,
        "budget": budget,
        "risk_filter": risk_filter,
        "channels": channels,
        "max_customers": max_customers,
        "operator_note": operator_note,
        "source_note": source_note,
    }
    st.markdown("</div>", unsafe_allow_html=True)
    return submit, input_df, config


def validate_inputs(input_df, config):
    errors = []
    if input_df.empty:
        errors.append("Input data is empty. Upload a CSV, use pipeline outputs, or switch to demo sample.")
    if not config["campaign_name"].strip():
        errors.append("Campaign name is required.")
    if not config["risk_filter"]:
        errors.append("Select at least one risk band.")
    if not config["channels"]:
        errors.append("Select at least one execution channel.")
    return errors


def run_processing_flow(input_df, config):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("2. Processing")
    st.caption("The app is normalizing customer records, scoring priorities, and preparing outreach recommendations.")

    progress = st.progress(0)
    status_box = st.empty()

    status_box.info("Step 1/3: Validating and preparing customer features")
    time.sleep(0.25)
    progress.progress(33)

    master_df, outcomes_df, mode_message = run_model_inference(input_df, config["channels"])
    if master_df is None or outcomes_df is None:
        master_df, outcomes_df = normalize_input_to_master(input_df, config["channels"])
    status_box.info("Step 2/3: Scoring campaign recommendations")
    time.sleep(0.3)
    progress.progress(66)

    work_df, outcome_df = process_campaign(master_df, outcomes_df, config)
    status_box.success("Step 3/3: Dashboard payload ready")
    progress.progress(100)
    st.caption(mode_message)

    st.markdown("</div>", unsafe_allow_html=True)
    return work_df, outcome_df


def render_dashboard(results_df, outcomes_df, config):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("3. Results Dashboard")
    st.caption("Interactive summary of campaign readiness, risk profile, and simulated retention outcomes.")

    total_customers = len(results_df)
    critical_count = int((results_df["risk_band"] == "critical_risk").sum())
    high_count = int((results_df["risk_band"] == "high_risk").sum())
    avg_churn = float(results_df["churn_score"].mean() * 100) if total_customers else 0.0
    outreach_share = float((outcomes_df["outreach_sent"].mean() * 100) if not outcomes_df.empty else 0.0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Customers in Scope", f"{total_customers:,}")
    m2.metric("Critical Risk", f"{critical_count:,}")
    m3.metric("High Risk", f"{high_count:,}")
    m4.metric("Avg Churn Score", f"{avg_churn:.1f}%")
    m5.metric("Outreach Coverage", f"{outreach_share:.1f}%")

    if critical_count > max(25, int(0.18 * total_customers)):
        st.markdown(
            "<div class='kpi-note'><strong>Alert:</strong> Critical-risk density is high. Consider increasing advisor calls and retention incentives for the first wave.</div>",
            unsafe_allow_html=True,
        )
    elif total_customers == 0:
        st.warning("No customers matched the current filters. Expand risk bands or channels to see results.")
    else:
        st.markdown(
            "<div class='kpi-note'><strong>Healthy Mix:</strong> Current targeting is balanced for outreach scale and expected save potential.</div>",
            unsafe_allow_html=True,
        )

    tab_overview, tab_insights, tab_actions, tab_export = st.tabs(["Overview", "Insights", "Action List", "Download"])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            risk_counts = results_df["risk_band"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
            risk_counts.columns = ["risk_band", "count"]
            fig_risk = px.pie(
                risk_counts,
                names="risk_band",
                values="count",
                hole=0.55,
                color="risk_band",
                color_discrete_map=RISK_COLORS,
                title="Risk Composition",
            )
            fig_risk.update_layout(margin=dict(t=50, b=10, l=10, r=10), legend_title=None)
            st.plotly_chart(fig_risk, use_container_width=True)

        with c2:
            hist = px.histogram(
                results_df,
                x="churn_score",
                nbins=20,
                color_discrete_sequence=["#0b6e4f"],
                title="Churn Score Distribution",
            )
            hist.update_layout(margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(hist, use_container_width=True)

    with tab_insights:
        c1, c2 = st.columns(2)
        with c1:
            reason_counts = results_df["top_reason_1"].value_counts().head(8).reset_index()
            reason_counts.columns = ["Reason", "Count"]
            fig_reason = px.bar(
                reason_counts.sort_values("Count"),
                x="Count",
                y="Reason",
                orientation="h",
                color="Count",
                color_continuous_scale="YlOrRd",
                title="Top Churn Drivers",
            )
            fig_reason.update_layout(margin=dict(t=50, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_reason, use_container_width=True)

        with c2:
            if outcomes_df.empty:
                st.info("No outcome records available for trend analysis yet.")
            else:
                trend = outcomes_df.copy()
                trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce").dt.date
                trend = trend.groupby("timestamp", as_index=False)["retained_90d"].mean()
                trend["retained_90d"] = trend["retained_90d"] * 100
                fig_trend = px.area(
                    trend,
                    x="timestamp",
                    y="retained_90d",
                    title="Simulated Retention Trend (%)",
                    color_discrete_sequence=["#2a9d8f"],
                )
                fig_trend.update_layout(margin=dict(t=50, b=10, l=10, r=10), yaxis_title="Retention %")
                st.plotly_chart(fig_trend, use_container_width=True)

        channel_perf = outcomes_df[outcomes_df["outreach_sent"] == True].groupby("execution_channel", as_index=False)["retained_90d"].mean()
        if not channel_perf.empty:
            channel_perf["retained_90d"] = channel_perf["retained_90d"] * 100
            fig_channel = px.bar(
                channel_perf,
                x="execution_channel",
                y="retained_90d",
                color="execution_channel",
                title="Retention by Channel (%)",
            )
            fig_channel.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_channel, use_container_width=True)

    with tab_actions:
        chips = []
        chips.append("<span class='status-chip chip-danger'>Critical: {}</span>".format(critical_count))
        chips.append("<span class='status-chip chip-warning'>High: {}</span>".format(high_count))
        chips.append("<span class='status-chip chip-success'>Budget: ${:,.0f}</span>".format(config["budget"]))
        st.markdown("".join(chips), unsafe_allow_html=True)

        display_cols = [
            "customer_id",
            "priority_score",
            "churn_score",
            "risk_band",
            "segment_name",
            "top_reason_1",
            "execution_channel",
            "chosen_action",
            "action_note",
        ]
        st.dataframe(
            results_df[display_cols].sort_values("priority_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Rows are ranked by campaign priority score and constrained by your selected limits.")

    with tab_export:
        st.info(
            f"Campaign: {config['campaign_name']} | Objective: {config['campaign_objective']} | Source: {config['source_note']}"
        )
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prioritized Action List",
            data=csv_bytes,
            file_name=f"{config['campaign_name'].strip().replace(' ', '_').lower()}_action_list.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(config["operator_note"])

    st.markdown("</div>", unsafe_allow_html=True)


def init_session_state():
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = None
    if "processed_outcomes" not in st.session_state:
        st.session_state.processed_outcomes = None
    if "last_campaign_cfg" not in st.session_state:
        st.session_state.last_campaign_cfg = None


def main():
    apply_custom_css()
    init_session_state()

    outputs_ready = _outputs_ready()
    output_master, output_outcomes = load_output_data()

    render_sidebar_controls(outputs_ready)
    render_header()

    submit, input_df, config = collect_user_input(output_master)

    if submit:
        input_errors = validate_inputs(input_df, config)
        if input_errors:
            for err in input_errors:
                st.error(err)
        else:
            results_df, outcomes_df = run_processing_flow(input_df, config)
            st.session_state.processed_results = results_df
            st.session_state.processed_outcomes = outcomes_df
            st.session_state.last_campaign_cfg = config

    if st.session_state.processed_results is not None:
        render_dashboard(
            st.session_state.processed_results,
            st.session_state.processed_outcomes,
            st.session_state.last_campaign_cfg,
        )
    else:
        # Default first-load dashboard with meaningful demo visuals.
        default_input = output_master if not output_master.empty else load_demo_base_data()
        default_cfg = {
            "campaign_name": "Demo Campaign",
            "campaign_objective": "Reduce churn",
            "run_date": pd.Timestamp.today().date(),
            "budget": 15000,
            "risk_filter": RISK_ORDER,
            "channels": CHANNEL_CHOICES,
            "max_customers": 200,
            "operator_note": "This is a default preview dashboard before input submission.",
            "source_note": "Auto demo preview",
        }
        default_master, default_outcomes = normalize_input_to_master(default_input, CHANNEL_CHOICES)
        default_results, default_outcome_view = process_campaign(default_master, default_outcomes, default_cfg)
        st.info("Preview mode is active. Configure inputs and click Process Campaign to run a custom scenario.")
        render_dashboard(default_results, default_outcome_view, default_cfg)


if __name__ == "__main__":
    main()
