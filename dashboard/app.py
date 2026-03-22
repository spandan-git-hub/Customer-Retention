import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.channel_model import ChannelModel
from src.models.churn_model import ChurnModel
from src.rules.action_engine import ActionEngine

st.set_page_config(
    layout="wide",
    page_title="Retention Decision Engine",
    page_icon="\U0001F6E1\uFE0F",
)

ROOT_DIR = PROJECT_ROOT
OUTPUT_DIR = ROOT_DIR / "outputs"
DEMO_CSV_PATH = CURRENT_DIR / "demo_input.csv"
FEEDBACK_LOG_PATH = OUTPUT_DIR / "feedback_loop_log.csv"

RISK_ORDER = ["critical_risk", "high_risk", "medium_risk", "low_risk"]
RISK_COLORS = {
    "critical_risk": "#D62839",
    "high_risk": "#F77F00",
    "medium_risk": "#FCBF49",
    "low_risk": "#2A9D8F",
}
CHANNEL_CHOICES = ["Email", "SMS", "Call", "Push", "In-App"]
EDITABLE_FEEDBACK_COLS = ["outreach_sent", "responded", "retained_90d"]


def _format_indian_number(number, decimals=0):
    sign = "-" if float(number) < 0 else ""
    number = abs(float(number))
    number = round(number, decimals)
    int_part = int(number)
    frac_part = f"{number:.{decimals}f}".split(".")[1] if decimals > 0 else ""

    s = str(int_part)
    if len(s) <= 3:
        grouped = s
    else:
        grouped = s[-3:]
        s = s[:-3]
        while len(s) > 2:
            grouped = s[-2:] + "," + grouped
            s = s[:-2]
        if s:
            grouped = s + "," + grouped

    if decimals > 0:
        return f"{sign}{grouped}.{frac_part}"
    return f"{sign}{grouped}"


def format_inr(value, decimals=0):
    return f"₹{_format_indian_number(value, decimals=decimals)}"


def format_ddmmyyyy(dt_value):
    if pd.isna(dt_value):
        return ""
    return pd.to_datetime(dt_value).strftime("%d/%m/%Y")


def format_date_for_filename(dt_value):
    return pd.to_datetime(dt_value).strftime("%d%m%Y")


def apply_custom_css_light():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

        :root {
            --bg-1: #f4f1ea;
            --bg-2: #fdfdfd;
            --ink: #1f2933;
            --muted: #52606d;
            --accent: #0b6e4f;
            --accent-soft: #e4f2ee;
            --radius-xl: 20px;
            --radius-md: 13px;
            --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.08);
            --card-bg: rgba(255, 255, 255, 0.95);
            --card-border: rgba(15, 23, 42, 0.08);
            --table-bg: rgba(255, 255, 255, 0.96);
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 4%, #fcefd6 0%, rgba(0, 0, 0, 0) 32%),
                radial-gradient(circle at 95% 22%, #dcebe4 0%, rgba(0, 0, 0, 0) 28%),
                linear-gradient(180deg, var(--bg-1), #f7f7f5 45%, var(--bg-2));
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--card-border);
        }

        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-xl);
            padding: 0.95rem;
            box-shadow: var(--shadow-soft);
            margin-bottom: 0.9rem;
        }

        .subtle-note {
            background: var(--accent-soft);
            border-left: 4px solid var(--accent);
            border-radius: 9px;
            padding: 0.65rem 0.85rem;
            color: var(--ink);
            font-size: 0.9rem;
            margin-bottom: 0.7rem;
        }

        .nav-note {
            margin-top: 0.6rem;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            background: var(--table-bg);
            padding: 0.6rem 0.65rem;
            font-size: 0.84rem;
            color: var(--muted);
            line-height: 1.35;
        }

        .trace-meta {
            font-size: 0.84rem;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .status-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 700;
            margin-left: 0.4rem;
        }

        .status-completed {
            background: #d1fae5;
            color: #065f46;
        }

        .status-warning {
            background: #ffedd5;
            color: #9a3412;
        }

        .status-failed {
            background: #fee2e2;
            color: #991b1b;
        }

        .empty-state {
            border: 1px dashed var(--card-border);
            background: var(--table-bg);
            border-radius: 16px;
            padding: 1.1rem;
            text-align: center;
        }

        .empty-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            margin-bottom: 0.35rem;
            color: var(--ink);
        }

        .empty-sub {
            color: var(--muted);
            margin: 0 auto;
            max-width: 760px;
            font-size: 0.92rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 11px;
            border: 1px solid #0b6e4f;
            background: #0b6e4f;
            color: #ffffff;
            font-weight: 700;
            transition: transform 120ms ease, background 150ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #09563e;
            border-color: #09563e;
            transform: translateY(-1px);
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus {
            outline: 3px solid rgba(11, 110, 79, 0.28);
            outline-offset: 1px;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-md);
            padding: 0.65rem;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMarkdownContainer"],
        .stCaption,
        label {
            color: var(--ink) !important;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stDateInput input,
        .stNumberInput input,
        [data-baseweb="select"] > div,
        [data-baseweb="tag"] {
            background: var(--table-bg) !important;
            color: var(--ink) !important;
            border-color: var(--card-border) !important;
        }

        [data-testid="stDataFrame"] {
            background: var(--table-bg);
            border-radius: 12px;
            border: 1px solid var(--card-border);
            overflow: hidden;
        }

        @media (max-width: 1024px) {
            .stTitle {
                font-size: 1.45rem;
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_demo_csv():
    if DEMO_CSV_PATH.exists():
        return pd.read_csv(DEMO_CSV_PATH)
    return pd.DataFrame()


def _load_model_config():
    config_path = ROOT_DIR / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _artifact_paths_from_config(config):
    artifact_folder = ROOT_DIR / config["paths"]["artifacts"]["folder"]
    return {
        "churn_model": artifact_folder / "churn_model.pkl",
        "churn_encoder": artifact_folder / "churn_encoders.pkl",
        "channel_model": artifact_folder / "channel_model.pkl",
        "channel_encoder": artifact_folder / "channel_encoders.pkl",
    }


def _artifacts_ready(config):
    return all(path.exists() for path in _artifact_paths_from_config(config).values())


def _channel_to_ui(value):
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


def _compute_changed_fields(before_df, after_df, limit=12):
    before_cols = set(before_df.columns)
    after_cols = set(after_df.columns)
    added = sorted(list(after_cols - before_cols))
    removed = sorted(list(before_cols - after_cols))

    changed = []
    common = [c for c in before_df.columns if c in after_cols]
    n = min(len(before_df), len(after_df), 40)
    if n > 0:
        b = before_df.iloc[:n].reset_index(drop=True)
        a = after_df.iloc[:n].reset_index(drop=True)
        for col in common:
            if not b[col].astype(str).equals(a[col].astype(str)):
                changed.append(col)

    fields = []
    if added:
        fields.extend([f"added: {c}" for c in added])
    if removed:
        fields.extend([f"removed: {c}" for c in removed])
    if changed:
        fields.extend([f"updated: {c}" for c in changed])
    return fields[:limit] if fields else ["No structural or sampled row changes detected."]


def make_stage_log(stage, purpose, before_df, after_df, status="completed", notes="", extra=None):
    return {
        "stage": stage,
        "purpose": purpose,
        "status": status,
        "input_shape": before_df.shape,
        "output_shape": after_df.shape,
        "fields_changed": _compute_changed_fields(before_df, after_df),
        "input_sample": before_df.head(8).copy(),
        "output_sample": after_df.head(8).copy(),
        "notes": notes,
        "extra": extra or {},
    }


def _derive_reason_labels(df):
    score = pd.to_numeric(df.get("churn_score", 0.4), errors="coerce").fillna(0.4)
    balance = pd.to_numeric(df.get("Total_Revolving_Bal", df.get("Balance", 0.0)), errors="coerce").fillna(0.0)
    inactive = pd.to_numeric(
        df.get("Months_Inactive_12_mon", df.get("Inactive_days", 0.0)),
        errors="coerce",
    ).fillna(0.0)

    reason = np.where(score >= 0.78, "declining_balance", "falling_spend_trend")
    reason = np.where(balance < balance.quantile(0.2), "credit_stress_signal", reason)
    reason = np.where(inactive > inactive.quantile(0.7), "low_account_activity", reason)
    reason = np.where((score >= 0.55) & (inactive > inactive.quantile(0.5)), "transaction_frequency_drop", reason)
    return pd.Series(reason, index=df.index)


def _normalize_feedback_value(value):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text = str(value).strip().lower()
    if text in ["yes", "true", "1"]:
        return "Yes"
    if text in ["no", "false", "0"]:
        return "No"
    return ""


def _apply_feedback_defaults(df):
    out = df.copy()
    for col in EDITABLE_FEEDBACK_COLS:
        if col not in out.columns:
            out[col] = "No"
        out[col] = out[col].map(_normalize_feedback_value)
        out[col] = out[col].replace("", "No")
    return out


def _create_feedback_log(outcomes_df, campaign_name, source_label, update_reason="manual_feedback_edit"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = outcomes_df[["customer_id", "execution_channel", "chosen_action", "outreach_sent", "responded", "retained_90d"]].copy()
    for col in EDITABLE_FEEDBACK_COLS:
        payload[col] = payload[col].map(_normalize_feedback_value)

    completed_mask = payload[EDITABLE_FEEDBACK_COLS].isin(["Yes", "No"]).all(axis=1)
    payload["campaign_name"] = campaign_name
    payload["source"] = source_label
    payload["update_reason"] = update_reason
    payload["logged_at"] = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")
    payload["retraining_eligible"] = completed_mask

    if FEEDBACK_LOG_PATH.exists():
        prev = pd.read_csv(FEEDBACK_LOG_PATH)
        full = pd.concat([prev, payload], ignore_index=True)
    else:
        full = payload
    full.to_csv(FEEDBACK_LOG_PATH, index=False)

    return {
        "feedback_rows_logged": int(len(payload)),
        "retraining_ready_rows": int(completed_mask.sum()),
        "feedback_store": str(FEEDBACK_LOG_PATH.name),
        "update_reason": update_reason,
    }


def run_pipeline_with_trace(df_input, config, source_label):
    logs = []
    df = df_input.copy()

    id_candidates = ["customer_id", "CLIENTNUM", "CustomerId", "customerId", "id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        df["customer_id"] = np.arange(100000, 100000 + len(df))
    else:
        df["customer_id"] = df[id_col]

    # 1) Segmentation model
    before = df.copy()
    balance = pd.to_numeric(df.get("Total_Revolving_Bal", df.get("Balance", 0)), errors="coerce").fillna(0)
    df["segment_name"] = np.where(balance > 100000, "premium_active", np.where(balance < 20000, "dormant_low_value", "at_risk_mid_tier"))
    logs.append(
        make_stage_log(
            stage="1. Segmentation model",
            purpose="Classify customers into business-value segments.",
            before_df=before,
            after_df=df,
            notes="Segment assignment derived from balance thresholds.",
        )
    )

    # 2) Behavioural drift model / sequence layer
    before = df.copy()
    inactive = pd.to_numeric(df.get("Months_Inactive_12_mon", 0), errors="coerce").fillna(0)
    rel_count = pd.to_numeric(df.get("Total_Relationship_Count", 0), errors="coerce").fillna(0)
    df["behaviour_drift_score"] = ((inactive * 0.65) + (1 / (rel_count + 1) * 10 * 0.35)).round(3)
    df["drift_band"] = pd.cut(
        df["behaviour_drift_score"],
        bins=[-0.1, 1.5, 3.5, 7.0, 100],
        labels=["stable", "watch", "drifting", "critical_drift"],
    ).astype(str)
    logs.append(
        make_stage_log(
            stage="2. Behavioural drift model / sequence layer",
            purpose="Detect behavior drift from inactivity and engagement proxies.",
            before_df=before,
            after_df=df,
            notes="Drift scoring is simulated using available behavioral fields.",
        )
    )

    # 3) Churn prediction model
    before = df.copy()
    config_model = _load_model_config()
    churn_mode = "simulated"
    if _artifacts_ready(config_model):
        try:
            artifacts = _artifact_paths_from_config(config_model)
            churn_model = ChurnModel()
            churn_model.load(str(artifacts["churn_model"]), str(artifacts["churn_encoder"]))
            scores = churn_model.predict_proba(df)
            churn_mode = "artifact-based"
        except Exception:
            bal_norm = (balance - balance.min()) / (balance.max() - balance.min() + 1e-9)
            in_norm = (inactive - inactive.min()) / (inactive.max() - inactive.min() + 1e-9)
            scores = (0.65 * in_norm + 0.35 * (1 - bal_norm)).clip(0, 1)
    else:
        bal_norm = (balance - balance.min()) / (balance.max() - balance.min() + 1e-9)
        in_norm = (inactive - inactive.min()) / (inactive.max() - inactive.min() + 1e-9)
        scores = (0.65 * in_norm + 0.35 * (1 - bal_norm)).clip(0, 1)

    df["churn_score"] = pd.to_numeric(scores, errors="coerce").fillna(0.4).clip(0, 1)
    thresholds = config_model["churn_model"]["thresholds"]
    df["risk_band"] = pd.cut(
        df["churn_score"],
        bins=[-np.inf, thresholds["low_risk"], thresholds["medium_risk"], thresholds["high_risk"], np.inf],
        labels=["low_risk", "medium_risk", "high_risk", "critical_risk"],
    ).astype(str)
    logs.append(
        make_stage_log(
            stage="3. Churn prediction model",
            purpose="Estimate churn probability and assign risk bands.",
            before_df=before,
            after_df=df,
            notes=f"Churn scoring mode: {churn_mode}",
            extra={"avg_churn_score": float(df['churn_score'].mean()) if len(df) else 0.0},
        )
    )

    # 4) Explainability module
    before = df.copy()
    df["top_reason_1"] = _derive_reason_labels(df)
    df["explainability_confidence"] = (0.62 + (df["churn_score"] * 0.3)).clip(0, 1).round(3)
    logs.append(
        make_stage_log(
            stage="4. Explainability module",
            purpose="Generate primary churn driver and confidence metadata.",
            before_df=before,
            after_df=df,
            notes="Attribution is approximated for dashboard-level traceability.",
        )
    )

    # 5) Channel selection model
    before = df.copy()
    channel_mode = "simulated"
    channel_ui = pd.Series(["Email"] * len(df), index=df.index)
    channel_confidence = pd.Series([0.55] * len(df), index=df.index)
    if _artifacts_ready(config_model):
        try:
            artifacts = _artifact_paths_from_config(config_model)
            channel_model = ChannelModel()
            channel_model.load(str(artifacts["channel_model"]), str(artifacts["channel_encoder"]))
            model_input = df.copy()
            model_input["segment_name_encoded"] = model_input["segment_name"]
            pred, conf = channel_model.predict(model_input)
            channel_ui = pd.Series(pred, index=df.index).map(_channel_to_ui)
            channel_confidence = pd.Series(conf, index=df.index)
            channel_mode = "artifact-based"
        except Exception:
            channel_ui = np.where(df["risk_band"].isin(["critical_risk", "high_risk"]), "Call", np.where(df["risk_band"] == "medium_risk", "SMS", "Email"))
            channel_ui = pd.Series(channel_ui, index=df.index)
            channel_confidence = (0.5 + df["churn_score"] * 0.2).clip(0, 1)
    else:
        channel_ui = np.where(df["risk_band"].isin(["critical_risk", "high_risk"]), "Call", np.where(df["risk_band"] == "medium_risk", "SMS", "Email"))
        channel_ui = pd.Series(channel_ui, index=df.index)
        channel_confidence = (0.5 + df["churn_score"] * 0.2).clip(0, 1)

    if config["channels"]:
        allowed = set(config["channels"])
        fallback = config["channels"][0]
        channel_ui = channel_ui.map(lambda c: c if c in allowed else fallback)

    df["execution_channel"] = channel_ui
    df["channel_confidence"] = pd.to_numeric(channel_confidence, errors="coerce").fillna(0.55).round(3)
    logs.append(
        make_stage_log(
            stage="5. Channel selection model",
            purpose="Recommend best channel and confidence for each customer.",
            before_df=before,
            after_df=df,
            notes=f"Channel selection mode: {channel_mode}",
        )
    )

    # 6) Next-best-action model
    before = df.copy()
    action_engine = ActionEngine(config_path=str(ROOT_DIR / "configs" / "rules_config.yaml"))
    model_df = df.copy()
    model_df["best_channel"] = model_df["execution_channel"].map(_channel_from_ui)
    action_values = model_df.apply(action_engine.determine_action, axis=1)
    action_values.columns = ["chosen_action", "action_priority", "action_note", "execution_channel_raw"]
    df = pd.concat([df, action_values[["chosen_action", "action_priority", "action_note"]]], axis=1)
    logs.append(
        make_stage_log(
            stage="6. Next-best-action model",
            purpose="Select intervention actions from risk and context features.",
            before_df=before,
            after_df=df,
            notes="Rule engine output includes action, priority, and execution note.",
        )
    )

    # Campaign filter + ranking
    results = df.copy()
    results = results[results["risk_band"].isin(config["risk_filter"])]
    results = results[results["execution_channel"].isin(config["channels"])]
    risk_weight = {"critical_risk": 1.0, "high_risk": 0.8, "medium_risk": 0.55, "low_risk": 0.3}
    results["priority_score"] = results["churn_score"].clip(0, 1) * results["risk_band"].map(risk_weight).fillna(0.5)
    results = results.sort_values("priority_score", ascending=False).head(config["max_customers"])

    # 7) Feedback loop logs
    before = results.copy()
    rng = np.random.default_rng(11)
    outcomes = results[["customer_id", "risk_band", "execution_channel", "chosen_action", "top_reason_1"]].copy()
    outcomes.rename(columns={"top_reason_1": "predicted_reason"}, inplace=True)
    outcomes["timestamp"] = pd.Timestamp.today().normalize() - pd.to_timedelta(rng.integers(0, 28, size=len(outcomes)), unit="d")
    outcomes["outreach_sent"] = "No"
    outcomes["responded"] = "No"
    outcomes["retained_90d"] = "No"
    feedback_meta = {
        "feedback_rows_logged": 0,
        "retraining_ready_rows": 0,
        "feedback_store": str(FEEDBACK_LOG_PATH.name),
        "update_reason": "awaiting_manual_feedback",
    }

    logs.append(
        make_stage_log(
            stage="7. Feedback loop model logs",
            purpose="Persist post-action outcomes as retraining data for channel model improvement.",
            before_df=before,
            after_df=outcomes,
            notes="Feedback labels are blank by default and must be entered manually before retraining storage.",
            extra=feedback_meta,
        )
    )

    return results, outcomes, logs


def validate_inputs(input_df, campaign_name, risk_filter, channels):
    errors = []
    if input_df.empty:
        errors.append("Input data is empty. Provide demo edits or upload a CSV file.")
    if not campaign_name.strip():
        errors.append("Campaign name is required.")
    if not risk_filter:
        errors.append("Select at least one risk band.")
    if not channels:
        errors.append("Select at least one execution channel.")
    return errors


def render_empty_state(title, message):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-title">{title}</div>
            <p class="empty-sub">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _feedback_rate(series):
    mapped = pd.Series(series).map(lambda x: {"Yes": 1.0, "No": 0.0}.get(_normalize_feedback_value(x), np.nan))
    return float(mapped.mean() * 100) if mapped.notna().any() else 0.0


def _render_summary_metrics(results_df, outcomes_df, budget):
    total_customers = len(results_df)
    critical_count = int((results_df["risk_band"] == "critical_risk").sum())
    high_count = int((results_df["risk_band"] == "high_risk").sum())
    avg_churn = float(results_df["churn_score"].mean() * 100) if total_customers else 0.0
    outreach_share = _feedback_rate(outcomes_df["outreach_sent"]) if not outcomes_df.empty else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Critical Risk", f"{critical_count:,}")
    c3.metric("High Risk", f"{high_count:,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg Churn", f"{avg_churn:.1f}%")
    c5.metric("Outreach Coverage", f"{outreach_share:.1f}%")
    c6.metric("Incentive Budget", format_inr(budget))


def render_stage_logs(logs):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Show Details: Stage-by-Stage Processing Logs")
    st.caption("Each stage includes input sample, output sample, transformations, and auditable change summary.")

    status_class = {
        "completed": "status-completed",
        "warning": "status-warning",
        "failed": "status-failed",
    }

    for item in logs:
        tag_class = status_class.get(item["status"], "status-completed")
        with st.expander(item["stage"], expanded=False):
            st.markdown(
                f"<span class='status-chip {tag_class}'>{item['status'].upper()}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Purpose:** {item['purpose']}")
            st.markdown(
                f"<div class='trace-meta'>Input shape: {item['input_shape']} | Output shape: {item['output_shape']}</div>",
                unsafe_allow_html=True,
            )

            if item["notes"]:
                st.info(item["notes"])

            st.markdown("**What changed from previous step**")
            for field in item["fields_changed"]:
                st.write(f"- {field}")

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Input sample")
                st.dataframe(item["input_sample"], width="stretch", hide_index=True)
            with c2:
                st.caption("Output sample")
                st.dataframe(item["output_sample"], width="stretch", hide_index=True)

            if item["extra"]:
                st.caption("Intermediate scores / metadata")
                st.json(item["extra"])

    st.markdown("</div>", unsafe_allow_html=True)


def _feedback_editor(flow_prefix, source_label):
    outcomes_key = f"{flow_prefix}_outcomes"
    logs_key = f"{flow_prefix}_logs"
    config_key = f"{flow_prefix}_config"

    outcomes_df = st.session_state.get(outcomes_key)
    if outcomes_df is None or outcomes_df.empty:
        st.info("No feedback loop data available for editing yet.")
        return

    st.caption("Enter feedback labels manually using Yes/No. You can update these values continuously over time.")

    view_df = _apply_feedback_defaults(outcomes_df)
    view_df["timestamp"] = pd.to_datetime(view_df["timestamp"], errors="coerce")
    view_df["date"] = view_df["timestamp"].map(format_ddmmyyyy)

    non_editable = [c for c in view_df.columns if c not in EDITABLE_FEEDBACK_COLS]
    edited = st.data_editor(
        view_df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        disabled=non_editable,
        key=f"{flow_prefix}_feedback_editor",
        column_config={
            "outreach_sent": st.column_config.SelectboxColumn("outreach_sent", options=["No", "Yes"]),
            "responded": st.column_config.SelectboxColumn("responded", options=["No", "Yes"]),
            "retained_90d": st.column_config.SelectboxColumn("retained_90d", options=["No", "Yes"]),
        },
    )

    if st.button("Save / Update Feedback", width="stretch", key=f"{flow_prefix}_apply_feedback"):
        updated_outcomes = outcomes_df.copy()
        edited = _apply_feedback_defaults(edited)
        after_values = edited[EDITABLE_FEEDBACK_COLS].applymap(_normalize_feedback_value).replace("", "No")

        for col in EDITABLE_FEEDBACK_COLS:
            updated_outcomes[col] = after_values[col]

        st.session_state[outcomes_key] = updated_outcomes

        # Append retraining save log into stage logs for transparency.
        retraining_ready = int(
            after_values.isin(["Yes", "No"]).all(axis=1).sum()
        )
        stage_log = make_stage_log(
            stage="7b. Feedback loop retraining edits",
            purpose="Capture current feedback labels for retraining channel selection model.",
            before_df=outcomes_df,
            after_df=updated_outcomes,
            notes="Current feedback table saved and logged for retraining simulation.",
            extra={"rows_in_feedback_table": int(len(updated_outcomes)), "retraining_ready_rows": retraining_ready},
        )
        st.session_state[logs_key] = (st.session_state.get(logs_key) or []) + [stage_log]

        feedback_meta = _create_feedback_log(
            updated_outcomes,
            campaign_name=st.session_state[config_key]["campaign_name"],
            source_label=source_label,
            update_reason="manual_feedback_edit",
        )

        st.success(
            f"Feedback data saved. Retraining-ready rows: {feedback_meta['retraining_ready_rows']}."
        )


def render_results_dashboard(results_df, outcomes_df, config, flow_prefix, source_label, allow_download=False):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Pipeline Output Dashboard")
    st.caption(
        f"Campaign: {config['campaign_name']} | Objective: {config['campaign_objective']} | Run Date: {format_ddmmyyyy(config['run_date'])}"
    )

    _render_summary_metrics(results_df, outcomes_df, config["budget"])

    if len(results_df) == 0:
        st.warning("No records matched current filters. Adjust risk bands, channels, or max customers.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)
    with col1:
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
            template="plotly_white",
        )
        fig_risk.update_layout(margin=dict(t=50, b=10, l=10, r=10), legend_title=None)
        st.plotly_chart(fig_risk, width="stretch")

    with col2:
        hist = px.histogram(
            results_df,
            x="churn_score",
            nbins=20,
            color_discrete_sequence=["#0b6e4f"],
            title="Churn Score Distribution",
            template="plotly_white",
        )
        hist.update_layout(margin=dict(t=50, b=10, l=10, r=10), xaxis_title="Churn Score", yaxis_title="Customers")
        st.plotly_chart(hist, width="stretch")

    action_df = results_df.copy()
    action_df["churn_score"] = (action_df["churn_score"] * 100).round(2)
    action_df["channel_confidence"] = (action_df.get("channel_confidence", 0.0) * 100).round(1)

    st.caption("Action recommendations")
    st.dataframe(
        action_df[
            [
                "customer_id",
                "priority_score",
                "churn_score",
                "risk_band",
                "segment_name",
                "top_reason_1",
                "execution_channel",
                "channel_confidence",
                "chosen_action",
                "action_note",
            ]
        ].sort_values("priority_score", ascending=False),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Feedback Loop / Retraining Data")
    st.caption("Edit response outcomes and save changes to simulate retraining data capture for channel model improvement.")
    _feedback_editor(flow_prefix=flow_prefix, source_label=source_label)

    if allow_download:
        export_df = results_df.copy()
        export_df["run_date"] = format_ddmmyyyy(config["run_date"])
        export_df["incentive_budget_inr"] = format_inr(config["budget"])
        file_name = f"{config['campaign_name'].strip().replace(' ', '_').lower()}_{format_date_for_filename(config['run_date'])}.csv"
        st.download_button(
            "Download Processed Output (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            width="stretch",
            key=f"{flow_prefix}_download",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def collect_campaign_controls(prefix, default_budget):
    c1, c2, c3 = st.columns(3)
    with c1:
        campaign_name = st.text_input("Campaign Name", value=f"{prefix} Campaign", key=f"{prefix}_campaign_name")
    with c2:
        budget = st.number_input("Incentive Budget (INR)", min_value=0, value=default_budget, step=5000, key=f"{prefix}_budget")
    with c3:
        run_date = st.date_input("Run Date", value=date.today(), key=f"{prefix}_run_date")
    st.caption(f"Selected run date: {format_ddmmyyyy(run_date)}")

    c4, c5, c6 = st.columns(3)
    with c4:
        objective = st.selectbox(
            "Objective",
            ["Reduce churn", "Increase engagement", "Boost high-value retention", "Win-back dormant users"],
            key=f"{prefix}_objective",
        )
    with c5:
        risk_filter = st.multiselect("Risk Bands", options=RISK_ORDER, default=RISK_ORDER[:3], key=f"{prefix}_risk")
    with c6:
        channels = st.multiselect("Channels", options=CHANNEL_CHOICES, default=["Email", "SMS", "Call"], key=f"{prefix}_channels")

    max_customers = st.slider("Max Customers", min_value=25, max_value=1000, value=250, step=25, key=f"{prefix}_max")

    return {
        "campaign_name": campaign_name,
        "campaign_objective": objective,
        "run_date": run_date,
        "budget": budget,
        "risk_filter": risk_filter,
        "channels": channels,
        "max_customers": max_customers,
    }


def render_demo_run():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Demo Run")
    st.caption("Transparent sandbox: edit demo CSV data, run the pipeline, and inspect every stage.")
    st.markdown(
        "<div class='subtle-note'>Edit demo rows, run processing, and audit each pipeline stage using Show Details.</div>",
        unsafe_allow_html=True,
    )

    demo_df = load_demo_csv()
    if demo_df.empty:
        st.error("Demo CSV is missing. Ensure dashboard/demo_input.csv exists.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.caption("Demo CSV in use")
    edited_demo = st.data_editor(demo_df, width="stretch", num_rows="dynamic", key="demo_editor")

    config = collect_campaign_controls(prefix="demo", default_budget=125000)
    run_clicked = st.button("Run Demo Pipeline", type="primary", width="stretch", key="run_demo")

    if run_clicked:
        errors = validate_inputs(edited_demo, config["campaign_name"], config["risk_filter"], config["channels"])
        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Executing demo pipeline..."):
                results_df, outcomes_df, logs = run_pipeline_with_trace(edited_demo, config, source_label="demo")
                st.session_state.demo_results = results_df
                st.session_state.demo_outcomes = outcomes_df
                st.session_state.demo_logs = logs
                st.session_state.demo_config = config
            st.success("Demo run completed.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("demo_results") is not None:
        show_details = st.toggle("Show Details", value=False, key="demo_show_details")
        if show_details:
            render_stage_logs(st.session_state.demo_logs)
        render_results_dashboard(
            st.session_state.demo_results,
            st.session_state.demo_outcomes,
            st.session_state.demo_config,
            flow_prefix="demo",
            source_label="demo",
            allow_download=False,
        )
    else:
        render_empty_state(
            "No demo run yet",
            "Edit the demo CSV, click Run Demo Pipeline, then enable Show Details to inspect each stage.",
        )


def render_real_app():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Real App")
    st.caption("Production workflow: upload a CSV, run full pipeline, inspect stage logs, and export output.")
    st.markdown(
        "<div class='subtle-note'>Real App accepts CSV upload only for a simpler, production-ready input flow.</div>",
        unsafe_allow_html=True,
    )

    upload = st.file_uploader("Upload customer CSV", type=["csv"], key="real_upload")
    input_df = pd.DataFrame()

    if upload is not None:
        try:
            input_df = pd.read_csv(upload)
            st.success(f"Loaded {len(input_df):,} rows from uploaded file.")
            st.dataframe(input_df.head(8), width="stretch")
        except Exception as ex:
            st.error("Unable to parse uploaded CSV.")
            st.exception(ex)

    config = collect_campaign_controls(prefix="real", default_budget=250000)
    run_clicked = st.button("Run Real Pipeline", type="primary", width="stretch", key="run_real")

    if run_clicked:
        errors = validate_inputs(input_df, config["campaign_name"], config["risk_filter"], config["channels"])
        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Executing real pipeline..."):
                results_df, outcomes_df, logs = run_pipeline_with_trace(input_df, config, source_label="real")
                st.session_state.real_results = results_df
                st.session_state.real_outcomes = outcomes_df
                st.session_state.real_logs = logs
                st.session_state.real_config = config
            st.success("Real pipeline completed.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("real_results") is not None:
        show_details = st.toggle("Show Details", value=False, key="real_show_details")
        if show_details:
            render_stage_logs(st.session_state.real_logs)
        render_results_dashboard(
            st.session_state.real_results,
            st.session_state.real_outcomes,
            st.session_state.real_config,
            flow_prefix="real",
            source_label="real",
            allow_download=True,
        )
    else:
        render_empty_state(
            "No real run yet",
            "Upload a CSV and click Run Real Pipeline to process data and view full pipeline transparency.",
        )


def render_sidebar_nav():
    with st.sidebar:
        st.markdown("### Navigation")
        selected = st.radio(
            "",
            ["Demo Run", "Real App"],
            index=0,
            key="nav_view",
            label_visibility="collapsed",
        )
        note = (
            "**Demo Run:** edit sandbox CSV and test stage behavior safely."
            if selected == "Demo Run"
            else "**Real App:** upload your real CSV, process it, and review auditable outputs."
        )
        st.markdown(f"<div class='nav-note'>{note}</div>", unsafe_allow_html=True)
        return selected


def init_session_state():
    defaults = {
        "demo_results": None,
        "demo_outcomes": None,
        "demo_logs": None,
        "demo_config": None,
        "real_results": None,
        "real_outcomes": None,
        "real_logs": None,
        "real_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    init_session_state()
    apply_custom_css_light()

    current_view = render_sidebar_nav()
    st.title("Retention Decision Engine")
    st.caption("Transparent retention analytics workspace")

    if current_view == "Demo Run":
        render_demo_run()
    else:
        render_real_app()


if __name__ == "__main__":
    main()
