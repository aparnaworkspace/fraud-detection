import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title = "Insurance Fraud Detector",
    page_icon  = "🔍",
    layout     = "wide"
)

# ── Load model + config ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/xgb_fraud_model.pkl')
    config = json.load(open('models/model_config.json'))
    return model, config

@st.cache_data
def load_data():
    df     = pd.read_csv('data/processed/features.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    return df, X_test, y_test

model, config   = load_model()
df, X_test, y_test = load_data()
THRESHOLD       = config['threshold']
FEATURE_NAMES   = config['feature_names']

# ── Header ───────────────────────────────────────────────────────
st.title("🔍 Health Insurance Fraud Detection")
st.markdown("**XGBoost + SHAP explainability · Trained on 558K+ Medicare claims · ROC-AUC 0.971**")
st.divider()

# ── Sidebar metrics ──────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("ROC-AUC",           "97.1%")
    st.metric("Recall (Fraud)",    "83.2%")
    st.metric("Precision (Fraud)", "66.1%")
    st.metric("Decision Threshold", THRESHOLD)
    st.metric("Training Claims",   "558K+")
    st.metric("Providers",         "5,410")
    st.divider()
    st.caption("CMS Medicare Provider Utilization Dataset")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔎 Provider Lookup", "📈 Model Insights", "📋 All Providers"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Provider Lookup
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Look up any provider")
    provider_list = df['Provider'].tolist()
    selected      = st.selectbox("Select Provider ID", provider_list)

    if selected:
        row      = df[df['Provider'] == selected].iloc[0]
        features = row[FEATURE_NAMES].values.reshape(1, -1)
        prob     = model.predict_proba(features)[0][1]
        pred     = int(prob >= THRESHOLD)
        actual   = int(row['PotentialFraud'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob*100:.1f}%")
        col2.metric("Prediction",
                    "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE",
                    delta=None)
        col3.metric("Actual Label",
                    "🚨 FRAUD" if actual == 1 else "✅ LEGITIMATE",
                    delta=None)

        st.divider()

        # Key stats
        st.subheader("Provider Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Claims",       int(row['TotalClaims']))
        c2.metric("Total Reimbursed",   f"${row['TotalReimbursed']:,.0f}")
        c3.metric("Unique Patients",    int(row['TotalUniquePatients']))
        c4.metric("Max Inpatient Stay", f"{row['IP_MaxLOS']:.0f} days")

        st.divider()

        # SHAP waterfall for this provider
        st.subheader("Why this prediction was made (SHAP)")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(
            pd.DataFrame([row[FEATURE_NAMES]], columns=FEATURE_NAMES)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values        = shap_values[0],
                base_values   = explainer.expected_value,
                data          = row[FEATURE_NAMES].values,
                feature_names = FEATURE_NAMES
            ),
            max_display = 12,
            show        = False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Charts")

    col1, col2 = st.columns(2)
    with col1:
        st.image('confusion_matrix_tuned.png',
                 caption='Confusion Matrix (Threshold 0.30)',
                 use_column_width=True)
    with col2:
        st.image('roc_auc_curve.png',
                 caption='ROC-AUC Curve (AUC = 0.971)',
                 use_column_width=True)

    st.divider()
    st.subheader("SHAP Feature Importance")
    col3, col4 = st.columns(2)
    with col3:
        st.image('shap_importance.png',
                 caption='Top 15 Features by SHAP Importance',
                 use_column_width=True)
    with col4:
        st.image('shap_beeswarm.png',
                 caption='SHAP Beeswarm — Feature Impact Direction',
                 use_column_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — All Providers Table
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("All Providers — Fraud Scores")

    # Score all providers
    X_all   = df[FEATURE_NAMES]
    probs   = model.predict_proba(X_all)[:, 1]
    preds   = (probs >= THRESHOLD).astype(int)

    display_df = pd.DataFrame({
        'Provider'          : df['Provider'],
        'Fraud Probability' : (probs * 100).round(1),
        'Prediction'        : ['🚨 FRAUD' if p == 1 else '✅ LEGITIMATE' for p in preds],
        'Actual'            : ['🚨 FRAUD' if a == 1 else '✅ LEGITIMATE' for a in df['PotentialFraud']],
        'Total Claims'      : df['TotalClaims'].astype(int),
        'Total Reimbursed'  : df['TotalReimbursed'].apply(lambda x: f"${x:,.0f}"),
        'Unique Patients'   : df['TotalUniquePatients'].astype(int),
    }).sort_values('Fraud Probability', ascending=False)

    # Filter
    filter_opt = st.radio("Filter", ["All", "Flagged Fraud Only", "Missed Fraud"],
                          horizontal=True)
    if filter_opt == "Flagged Fraud Only":
        display_df = display_df[display_df['Prediction'] == '🚨 FRAUD']
    elif filter_opt == "Missed Fraud":
        display_df = display_df[
            (display_df['Actual'] == '🚨 FRAUD') &
            (display_df['Prediction'] == '✅ LEGITIMATE')
        ]

    st.dataframe(display_df, use_container_width=True, height=500)
    st.caption(f"Showing {len(display_df)} providers")