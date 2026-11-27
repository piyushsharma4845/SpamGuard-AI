
import streamlit as st
import sys
import pandas as pd
import json
import sqlite3

from pathlib import Path
from datetime import datetime

# =====================================================
# PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "spamguard.db"

sys.path.insert(0, str(BASE_DIR / "src"))

from explainability import explain_email


st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# DATABASE
# =====================================================

def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_subject TEXT,
                prediction TEXT,
                confidence REAL,
                ham_probability REAL,
                spam_probability REAL,
                analyzed_at TEXT
            )
        """)


def save_prediction(subject, result):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO prediction_history
            (email_subject, prediction, confidence,
             ham_probability, spam_probability, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            subject if subject else "No Subject",
            result["prediction"],
            result["confidence"] * 100,
            result["probabilities"]["ham"] * 100,
            result["probabilities"]["spam"] * 100,
            datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
        ))


def get_history():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY id DESC",
            conn
        )


initialize_database()


# =====================================================
# LOAD MODEL METRICS
# =====================================================

metrics_path = BASE_DIR / "models" / "model_metrics.json"

with open(metrics_path, "r") as file:
    metrics = json.load(file)

best_model = metrics["best_model"]
accuracy = metrics["results"][best_model]["accuracy"] * 100


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080e1c 0%, #0c1629 55%, #101b30 100%);
    color: #e2e8f0;
}

[data-testid="stHeader"] {
    background: rgba(8, 14, 28, 0.85);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #0b1424;
    border-right: 1px solid #1e3048;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #a8b7cc;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 9px 10px;
    border-radius: 9px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #172941;
}

/* Typography */

h1, h2, h3 {
    color: #f1f5f9 !important;
    letter-spacing: -0.5px;
}

p, label {
    color: #cbd5e1;
}

.page-title {
    font-size: 35px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Brand */

.brand {
    font-size: 23px;
    font-weight: 800;
    color: #f8fafc;
}

.brand span {
    color: #38bdf8;
}

.brand-caption {
    color: #8495ad;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric Cards */

.metric-card {
    background: linear-gradient(145deg, #172840, #111e32);
    border: 1px solid #263c58;
    border-radius: 15px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.12);
}

.metric-label {
    color: #9aacc3;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #f8fafc;
    font-size: 29px;
    font-weight: 800;
}

.metric-note {
    color: #7188a6;
    font-size: 11px;
    margin-top: 5px;
}

/* Scanner Panel */

.scanner-panel {
    background: linear-gradient(145deg, #14243a, #101c2e);
    border: 1px solid #263e5c;
    padding: 23px;
    border-radius: 17px;
    margin-bottom: 20px;
}

/* Result Panels */

.result-spam {
    background: linear-gradient(120deg, #3b1724, #241827);
    border: 1px solid #9f354c;
    border-radius: 15px;
    padding: 22px;
}

.result-ham {
    background: linear-gradient(120deg, #12382f, #142b2c);
    border: 1px solid #287b62;
    border-radius: 15px;
    padding: 22px;
}

.result-title {
    font-size: 23px;
    font-weight: 800;
    color: #ffffff;
}

.result-caption {
    color: #c0cedd;
    font-size: 13px;
    margin-top: 6px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(90deg, #087eae, #2563eb);
    color: white;
    border: 1px solid #2688c8;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0e9acb, #3479f5);
    border-color: #38bdf8;
    color: white;
    transform: translateY(-1px);
}

/* Download Button */

.stDownloadButton > button {
    background: #142b42;
    color: #e0f2fe;
    border: 1px solid #2878a8;
    border-radius: 10px;
    min-height: 43px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background: #1d4565;
    color: white;
    border-color: #38bdf8;
}

/* Inputs */

.stTextInput input, .stTextArea textarea {
    background: #101d30;
    color: #f1f5f9;
    border: 1px solid #304660;
    border-radius: 9px;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 1px #38bdf8;
}

/* Select Boxes */

.stSelectbox div[data-baseweb="select"] > div {
    background: #101d30;
    border-color: #304660;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    border: 1px solid #263c58;
    border-radius: 10px;
}

/* Dividers */

hr {
    border-color: #26384f;
}

/* Progress Bar */

.stProgress > div > div > div {
    background: linear-gradient(90deg, #0891b2, #3b82f6);
}

/* Hide Streamlit Branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# REUSABLE UI FUNCTIONS
# =====================================================

def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, note=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    """, unsafe_allow_html=True)


# =====================================================
# SHOW SCAN RESULT
# =====================================================

def show_result(result, subject, body):

    prediction = result["prediction"]
    confidence = result["confidence"] * 100

    ham_probability = result["probabilities"]["ham"] * 100
    spam_probability = result["probabilities"]["spam"] * 100

    st.markdown("---")
    st.markdown("### 🛡️ Scan Results")

    if prediction == "spam":

        st.markdown("""
            <div class="result-spam">
                <div class="result-title">🚨 SPAM EMAIL DETECTED</div>
                <div class="result-caption">
                    The machine learning model classified this email as spam.
                    Treat its links and requests with caution.
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
            <div class="result-ham">
                <div class="result-title">✅ LEGITIMATE EMAIL</div>
                <div class="result-caption">
                    The machine learning model classified this email as legitimate.
                    This does not guarantee that the email is completely safe.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "PREDICTION CONFIDENCE",
            f"{confidence:.2f}%",
            "Model probability"
        )

    with col2:
        metric_card(
            "SPAM PROBABILITY",
            f"{spam_probability:.2f}%",
            "Spam classification"
        )

    with col3:
        metric_card(
            "LEGITIMATE PROBABILITY",
            f"{ham_probability:.2f}%",
            "Ham classification"
        )

    st.write("")

    # Classification Breakdown

    st.markdown("### 📊 Classification Breakdown")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.markdown("**Legitimate Email**")
        st.progress(ham_probability / 100)
        st.caption(f"{ham_probability:.2f}%")

    with prob_col2:
        st.markdown("**Spam Email**")
        st.progress(spam_probability / 100)
        st.caption(f"{spam_probability:.2f}%")

    # =================================================
    # LIME EXPLANATION
    # =================================================

    st.markdown("---")
    st.markdown("### 🧠 Explainable AI Analysis")

    st.caption(
        "LIME identifies words that influenced this individual prediction."
    )

    explanation_df = pd.DataFrame(
        result["important_words"],
        columns=["Word", "Contribution"]
    )

    explanation_df["Influence"] = explanation_df["Contribution"].apply(
        lambda x: "Supports Spam" if x > 0 else "Supports Ham"
    )

    explanation_df["Contribution"] = (
        explanation_df["Contribution"].round(4)
    )

    # Display formatted explanation table

    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Word": st.column_config.TextColumn(
                "Important Word"
            ),
            "Contribution": st.column_config.NumberColumn(
                "Contribution Score",
                format="%.4f"
            ),
            "Influence": st.column_config.TextColumn(
                "Model Influence"
            )
        }
    )

    # Copy-friendly explanation

    st.markdown("#### 📋 Copy Explanation")

    copy_text = explanation_df.to_string(index=False)

    st.caption(
        "Use the copy icon in the top-right corner of the text block."
    )

    st.code(copy_text, language=None)

    # CSV Download

    st.download_button(
        label="⬇️ Download LIME Explanation (CSV)",
        data=explanation_df.to_csv(index=False).encode("utf-8"),
        file_name="spamguard_lime_explanation.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.info(
        "Positive contributions support spam classification. "
        "Negative contributions support legitimate classification. "
        "These are explanation scores, not probabilities."
    )

    # =================================================
    # SUSPICIOUS PHRASE DETECTION
    # =================================================

    st.markdown("---")
    st.markdown("### 🔎 Suspicious Phrase Detection")

    suspicious_phrases = [
        "free",
        "winner",
        "won",
        "claim now",
        "urgent",
        "click here",
        "congratulations",
        "limited offer",
        "verify your account",
        "password",
        "account suspended",
        "act now"
    ]

    email_lower = (subject + " " + body).lower()

    found_phrases = [
        phrase for phrase in suspicious_phrases
        if phrase in email_lower
    ]

    if found_phrases:

        st.warning(
            "Potentially suspicious phrases detected: "
            + ", ".join(found_phrases)
        )

    else:

        st.success(
            "No phrases from the basic suspicious-word list were detected."
        )


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("""
        <div style="padding: 8px 0 18px 0;">
            <div style="font-size:32px;">🛡️</div>
            <div class="brand">SPAMGUARD <span>AI</span></div>
            <div class="brand-caption">Intelligent Email Security</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("##### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Email Scanner",
            "Prediction History"
        ],
        format_func=lambda x: {
            "Dashboard": "🏠   Dashboard",
            "Email Scanner": "🔍   Email Scanner",
            "Prediction History": "📜   Prediction History"
        }[x],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("##### SYSTEM STATUS")

    st.markdown("""
        <div style="
            background:#142b2c;
            border:1px solid #286452;
            padding:12px;
            border-radius:10px;">
            <span style="color:#4ade80;">●</span>
            <span style="color:#d1fae5;font-size:13px;">
                Application Active
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption("Machine Learning • NLP • Explainable AI")
    st.caption("SpamGuard AI | Project Edition")


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    page_header(
        "Security Dashboard",
        "Monitor email threats and explore your SpamGuard AI activity."
    )

    history = get_history()

    total = len(history)

    spam_count = (
        len(history[history["prediction"] == "spam"])
        if total else 0
    )

    ham_count = (
        len(history[history["prediction"] == "ham"])
        if total else 0
    )

    # Welcome section

    st.markdown("""
        <div class="scanner-panel">
            <div style="font-size:13px;color:#38bdf8;font-weight:700;">
                SPAMGUARD INTELLIGENCE CENTER
            </div>
            <div style="font-size:25px;font-weight:800;color:#f8fafc;margin-top:8px;">
                Stay one step ahead of email threats.
            </div>
            <div style="color:#94a3b8;font-size:14px;margin-top:7px;">
                Analyse messages, detect spam patterns and understand
                AI-powered predictions.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "MODEL ACCURACY",
            f"{accuracy:.2f}%",
            best_model
        )

    with c2:
        metric_card(
            "TRAINING EMAILS",
            f'{metrics["training_samples"]:,}',
            "Model training data"
        )

    with c3:
        metric_card(
            "TEST EMAILS",
            f'{metrics["testing_samples"]:,}',
            "Model evaluation data"
        )

    st.write("")
    st.markdown("### 📬 Email Analysis Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("TOTAL ANALYSED", f"{total:,}", "Emails scanned")

    with c2:
        metric_card("SPAM DETECTED", f"{spam_count:,}", "Classified as spam")

    with c3:
        metric_card("LEGITIMATE EMAILS", f"{ham_count:,}", "Classified as ham")

    st.write("")

    if total > 0:

        chart_col, summary_col = st.columns([1.5, 1])

        with chart_col:

            st.markdown("### Classification Distribution")

            chart_data = pd.DataFrame({
                "Category": ["Spam", "Legitimate"],
                "Emails": [spam_count, ham_count]
            })

            st.bar_chart(
                chart_data.set_index("Category"),
                color="#22a6d5"
            )

        with summary_col:

            st.markdown("### Threat Summary")

            spam_percentage = (spam_count / total) * 100
            ham_percentage = (ham_count / total) * 100

            st.markdown("**Spam Percentage**")
            st.progress(spam_percentage / 100)
            st.caption(f"{spam_percentage:.1f}% of analysed emails")

            st.markdown("**Legitimate Percentage**")
            st.progress(ham_percentage / 100)
            st.caption(f"{ham_percentage:.1f}% of analysed emails")

        st.markdown("---")
        st.markdown("### 🕒 Recent Activity")

        recent = history.head(5).copy()

        recent["confidence"] = recent["confidence"].map(
            lambda x: f"{x:.2f}%"
        )

        st.dataframe(
            recent[
                ["email_subject", "prediction", "confidence", "analyzed_at"]
            ].rename(columns={
                "email_subject": "Email Subject",
                "prediction": "Prediction",
                "confidence": "Confidence",
                "analyzed_at": "Date and Time"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No emails analysed yet. Open Email Scanner to perform your first scan."
        )

    st.markdown("---")
    st.caption("SpamGuard AI | ML + NLP + Explainable AI")


# =====================================================
# EMAIL SCANNER
# =====================================================

elif page == "Email Scanner":

    page_header(
        "Email Security Scanner",
        "Scan email content using machine learning and explainable AI."
    )

    st.markdown("""
        <div class="scanner-panel">
            <div style="font-size:21px;font-weight:800;color:#f8fafc;">
                🔍 Intelligent Email Analysis
            </div>
            <div style="font-size:13px;color:#94a3b8;margin-top:6px;">
                Enter an email below to identify spam patterns and
                understand the model's decision.
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown("### 📧 Email Details")

        subject = st.text_input(
            "Email Subject",
            placeholder="Example: Congratulations! You won a prize"
        )

        body = st.text_area(
            "Email Content",
            placeholder="Paste the complete email message here...",
            height=230
        )

        st.caption(
            "The subject and message content are used for classification."
        )

        scan_button = st.button(
            "🛡️  SCAN EMAIL NOW",
            use_container_width=True
        )

    if scan_button:

        email_text = (subject + " " + body).strip()

        if not email_text:

            st.warning(
                "Please enter an email subject or message first."
            )

        else:

            with st.spinner(
                "Analysing email patterns and generating explanation..."
            ):

                try:

                    result = explain_email(email_text)

                    save_prediction(subject, result)

                    st.session_state["latest_result"] = {
                        "result": result,
                        "subject": subject,
                        "body": body
                    }

                    st.success(
                        "Scan completed. Result saved to prediction history."
                    )

                except Exception as error:

                    st.error(f"Analysis failed: {error}")

                    st.info(
                        "Check that the trained model and LIME files exist."
                    )

    # Keep results visible after Streamlit reruns

    if "latest_result" in st.session_state:

        latest = st.session_state["latest_result"]

        show_result(
            latest["result"],
            latest["subject"],
            latest["body"]
        )

        st.write("")

        if st.button("🗑️ Clear Current Result"):

            del st.session_state["latest_result"]
            st.rerun()

    st.markdown("---")
    st.caption("SpamGuard AI | Intelligent Email Security")


# =====================================================
# PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    page_header(
        "Prediction History",
        "Explore, search and review your previously analysed emails."
    )

    history = get_history()

    if history.empty:

        st.info(
            "Your history is currently empty. Scan an email to see it here."
        )

    else:

        total = len(history)

        spam_count = len(
            history[history["prediction"] == "spam"]
        )

        ham_count = len(
            history[history["prediction"] == "ham"]
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "TOTAL ANALYSES",
                f"{total:,}",
                "All saved predictions"
            )

        with c2:
            metric_card(
                "SPAM EMAILS",
                f"{spam_count:,}",
                "Spam classifications"
            )

        with c3:
            metric_card(
                "LEGITIMATE EMAILS",
                f"{ham_count:,}",
                "Ham classifications"
            )

        st.write("")
        st.markdown("### 🔎 Search and Filter")

        filter_col1, filter_col2 = st.columns([1, 1])

        with filter_col1:

            search = st.text_input(
                "Search email subject",
                placeholder="Enter a subject keyword..."
            )

        with filter_col2:

            prediction_filter = st.selectbox(
                "Filter by prediction",
                ["All", "Spam", "Legitimate"]
            )

        filtered = history.copy()

        if search:

            filtered = filtered[
                filtered["email_subject"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        if prediction_filter == "Spam":

            filtered = filtered[
                filtered["prediction"] == "spam"
            ]

        elif prediction_filter == "Legitimate":

            filtered = filtered[
                filtered["prediction"] == "ham"
            ]

        st.markdown("---")
        st.markdown("### 📋 Analysis Records")

        st.caption(
            f"Showing {len(filtered)} of {total} records"
        )

        display_history = filtered.rename(columns={
            "id": "ID",
            "email_subject": "Email Subject",
            "prediction": "Prediction",
            "confidence": "Confidence (%)",
            "ham_probability": "Legitimate (%)",
            "spam_probability": "Spam (%)",
            "analyzed_at": "Date and Time"
        })

        columns_to_show = [
            "ID",
            "Email Subject",
            "Prediction",
            "Confidence (%)",
            "Legitimate (%)",
            "Spam (%)",
            "Date and Time"
        ]

        st.dataframe(
            display_history[columns_to_show],
            use_container_width=True,
            hide_index=True
        )

        # CSV Export

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Export History as CSV",
            data=csv_data,
            file_name="spamguard_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")
        st.caption(
            "Prediction records are stored locally using SQLite."
        )
