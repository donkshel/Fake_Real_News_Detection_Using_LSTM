import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import string
import re
import nltk
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 18px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 0.5rem;
    letter-spacing: -0.5px;
}
.hero p { color: #b0aed8; font-size: 1.05rem; margin: 0; line-height: 1.7; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: #c9c3f5;
    border-radius: 50px;
    font-size: 0.78rem;
    padding: 4px 14px;
    margin-bottom: 1rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 700;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #4a4a7a;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* ── Verdict cards ── */
.verdict-real {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    border-left: 5px solid #52b788;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    color: white;
    box-shadow: 0 4px 20px rgba(82,183,136,0.25);
}
.verdict-fake {
    background: linear-gradient(135deg, #6b1a1a, #9b2226);
    border-left: 5px solid #e63946;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    color: white;
    box-shadow: 0 4px 20px rgba(230,57,70,0.25);
}
.verdict-uncertain {
    background: linear-gradient(135deg, #4a3800, #7a5f00);
    border-left: 5px solid #f4a261;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    color: white;
    box-shadow: 0 4px 20px rgba(244,162,97,0.25);
}
.verdict-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.verdict-sub { font-size: 0.95rem; opacity: 0.85; }

/* ── Stat boxes ── */
.stat-box {
    background: #f8f9fb;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    border: 1px solid #e2e5ec;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #1d1d3b;
}
.stat-label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Scorecard ── */
.score-card {
    background: linear-gradient(135deg, #1c1b3a, #2d2b55);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1rem;
}
.score-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #9b97cc;
    margin-bottom: 0.3rem;
}
.score-card-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.score-card-sub {
    font-size: 0.8rem;
    color: #9b97cc;
    margin-top: 0.2rem;
}
.score-badge-good   { color: #52b788; }
.score-badge-warn   { color: #f4a261; }
.score-badge-danger { color: #e63946; }

/* ── CM cells ── */
.cm-table { width: 100%; border-collapse: collapse; text-align: center; }
.cm-table td {
    padding: 1rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    border-radius: 8px;
}
.cm-tp { background: rgba(82,183,136,0.2);  color: #1b7a4e; }
.cm-tn { background: rgba(74,144,217,0.2);  color: #1a5fa0; }
.cm-fp { background: rgba(244,162,97,0.2);  color: #c4620a; }
.cm-fn { background: rgba(230,57,70,0.2);   color: #9b2226; }

/* ── Insight box ── */
.insight-box {
    background: #f0f3ff;
    border: 1px solid #c5cfff;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    color: #3a3a7a;
    margin-bottom: 0.7rem;
    line-height: 1.6;
}

/* ── Info box ── */
.info-box {
    background: #f0f3ff;
    border: 1px solid #c5cfff;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    color: #3a3a7a;
    margin-top: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #302b63, #24243e);
    color: white !important;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.7rem 2rem;
    border-radius: 10px;
    border: none;
    width: 100%;
    transition: all 0.25s ease;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4a4490, #302b63);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(48,43,99,0.35);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    if os.path.exists('fake_true_news_lstm_model.keras'):
        model = load_model('fake_true_news_lstm_model.keras')
    elif os.path.exists('fake_true_news_lstm_model.h5'):
        model = load_model('fake_true_news_lstm_model.h5')
    else:
        st.error("Model file not found.")
        st.stop()

    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    maxlen = 531
    if os.path.exists('model_config.json'):
        with open('model_config.json') as f:
            maxlen = json.load(f).get('smart_maxlen', 531)

    return model, tokenizer, maxlen

@st.cache_resource
def load_stopwords():
    for pkg in ['stopwords', 'punkt', 'punkt_tab']:
        nltk.download(pkg, quiet=True)
    sw = set(stopwords.words('english'))
    sw.update(['from', 'subject', 're', 'use'])
    return sw

@st.cache_data
def load_eval_metrics():
    if os.path.exists('eval_metrics.json'):
        with open('eval_metrics.json') as f:
            return json.load(f)
    return None

model, tokenizer, MAXLEN = load_resources()
stop_words = load_stopwords()
eval_data  = load_eval_metrics()


# ─────────────────────────────────────────────
# PREPROCESSING AS DONE IN TRAINING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text)
    text = re.sub(r'^[a-z\s/]+\s-\s', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join([w for w in text.split() if w not in stop_words])


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(text: str) -> dict | None:
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return None
    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')
    raw    = float(model.predict(padded, verbose=0)[0][0])
    real_p = raw * 100
    fake_p = (1 - raw) * 100
    label  = 'REAL' if raw > 0.7 else ('FAKE' if raw < 0.3 else 'UNCERTAIN')
    return {'label': label, 'raw': raw,
            'real_prob': real_p, 'fake_prob': fake_p,
            'word_count': len(cleaned.split()), 'cleaned': cleaned}


# ─────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────
def make_gauge(real_prob: float) -> go.Figure:
    color = '#52b788' if real_prob > 70 else ('#e63946' if real_prob < 30 else '#f4a261')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=real_prob,
        number={'suffix': '%', 'font': {'size': 38, 'family': 'Syne', 'color': color}},
        title={'text': "Authenticity Score",
               'font': {'size': 14, 'color': '#666', 'family': 'DM Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ccc',
                     'tickfont': {'size': 11}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#f4f4f8', 'borderwidth': 0,
            'steps': [
                {'range': [0,  30], 'color': 'rgba(230,57,70,0.12)'},
                {'range': [30, 70], 'color': 'rgba(244,162,97,0.12)'},
                {'range': [70, 100],'color': 'rgba(82,183,136,0.12)'},
            ],
            'threshold': {'line': {'color': color, 'width': 4},
                          'thickness': 0.8, 'value': real_prob}
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=10),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def make_roc_chart(e: dict) -> go.Figure:
    fpr = e['roc_fpr']; tpr = e['roc_tpr']
    roc_auc = e['roc_auc']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             line=dict(color='#aaa', dash='dash', width=1.5),
                             name='Random (AUC = 0.50)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             line=dict(color='#6c63ff', width=3),
                             fill='tozeroy', fillcolor='rgba(108,99,255,0.12)',
                             name=f'LSTM (AUC = {roc_auc:.4f})'))
    fig.update_layout(
        title='ROC Curve',
        xaxis=dict(title='False Positive Rate', range=[0,1]),
        yaxis=dict(title='True Positive Rate',  range=[0,1.02]),
        height=400, paper_bgcolor='white', plot_bgcolor='#f9f9fc',
        margin=dict(l=30, r=20, t=50, b=40),
        legend=dict(x=0.5, y=0.1, bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd', borderwidth=1),
    )
    return fig


def make_pr_chart(e: dict) -> go.Figure:
    prec = e['pr_precision']; rec = e['pr_recall']
    ap   = e['average_precision']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0.5,0.5], mode='lines',
                             line=dict(color='#aaa', dash='dash', width=1.5),
                             name='Random', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines',
                             line=dict(color='#52b788', width=3),
                             fill='tozeroy', fillcolor='rgba(82,183,136,0.12)',
                             name=f'LSTM (AP = {ap:.4f})'))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis=dict(title='Recall',    range=[0,1]),
        yaxis=dict(title='Precision', range=[0,1.02]),
        height=400, paper_bgcolor='white', plot_bgcolor='#f9f9fc',
        margin=dict(l=30, r=20, t=50, b=40),
        legend=dict(x=0.05, y=0.1, bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd', borderwidth=1),
    )
    return fig


def make_threshold_chart(e: dict, selected_threshold: float) -> go.Figure:
    tx   = e['thresh_x']
    data = [
        (e['thresh_acc'],  'Accuracy',  '#4a90d9'),
        (e['thresh_prec'], 'Precision', '#f4a261'),
        (e['thresh_rec'],  'Recall',    '#e63946'),
        (e['thresh_f1'],   'F1-Score',  '#52b788'),
    ]
    fig = go.Figure()
    for vals, name, color in data:
        fig.add_trace(go.Scatter(x=tx, y=vals, mode='lines', name=name,
                                 line=dict(color=color, width=2.5)))
    fig.add_vline(x=0.5, line_dash='dash', line_color='#888',
                  annotation_text='Default (0.5)', annotation_position='top right')
    fig.add_vline(x=e['optimal_threshold_f1'], line_dash='dot', line_color='#52b788',
                  annotation_text=f"Best F1 ({e['optimal_threshold_f1']:.2f})",
                  annotation_position='top left')
    if selected_threshold not in [0.5, e['optimal_threshold_f1']]:
        fig.add_vline(x=selected_threshold, line_dash='solid', line_color='#6c63ff',
                      annotation_text=f'Selected ({selected_threshold:.2f})',
                      annotation_position='bottom right')
    fig.update_layout(
        title='Threshold Sensitivity — How Metrics Change with Decision Threshold',
        xaxis=dict(title='Decision Threshold', range=[0,1]),
        yaxis=dict(title='Score',              range=[0,1.05]),
        height=400, paper_bgcolor='white', plot_bgcolor='#f9f9fc',
        hovermode='x unified',
        margin=dict(l=30, r=20, t=60, b=60),
        legend=dict(orientation='h', y=-0.22),
    )
    return fig


def make_cm_chart(e: dict) -> go.Figure:
    tp, tn, fp, fn = e['tp'], e['tn'], e['fp'], e['fn']
    z    = [[tn, fp], [fn, tp]]
    text = [[f'TN\n{tn}', f'FP\n{fp}'], [f'FN\n{fn}', f'TP\n{tp}']]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=['Predicted: Fake', 'Predicted: Real'],
        y=['Actual: Fake', 'Actual: Real'],
        text=text, texttemplate='%{text}',
        colorscale=[[0,'#f0f4ff'],[0.5,'#a5b4fc'],[1,'#4338ca']],
        showscale=False,
        textfont=dict(size=18, family='Syne'),
    ))
    fig.update_layout(
        title='Confusion Matrix (threshold = 0.5)',
        height=360, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(side='bottom'),
    )
    return fig


def make_class_bar(e: dict) -> go.Figure:
    metrics = ['precision', 'recall', 'f1-score']
    keys    = {
        'precision': ('precision_fake_at_0.5', 'precision_real_at_0.5'),
        'recall':    ('recall_fake_at_0.5',    'recall_real_at_0.5'),
        'f1-score':  ('f1_fake_at_0.5',        'f1_real_at_0.5'),
    }
    colors  = ['#e63946', '#52b788', '#4a90d9']
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fk, rk = keys[metric]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=['Fake', 'Real'],
            y=[e[fk], e[rk]],
            marker_color=color,
            text=[f"{e[fk]:.3f}", f"{e[rk]:.3f}"],
            textposition='outside',
        ))
    fig.update_layout(
        title='Per-Class Metrics — Precision, Recall, F1',
        barmode='group',
        yaxis=dict(title='Score', range=[0, 1.12]),
        xaxis=dict(title='Class'),
        height=380, paper_bgcolor='white', plot_bgcolor='#f9f9fc',
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation='h', y=-0.2),
    )
    return fig


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ LSTM · NLP · Deep Learning</div>
    <h1>🔍 Fake News Detection System</h1>
    <p>Paste a news article or upload a file — our LSTM model<br>
    will analyse it and estimate its authenticity in seconds.</p>
</div>
""", unsafe_allow_html=True)
st.divider()
st.write("This app uses a Bidirectional LSTM model trained on the Kaggle Fake and True News Dataset." \
" The model was trained to classify news articles as REAL or FAKE based on their text content. " \
    " Use the tabs below 👇 to explore the model's performance and the dataset used for training.")

# ─────────────────────────────────────────────
# TOP-LEVEL NAVIGATION TABS
# ─────────────────────────────────────────────
page_detect, page_eval, page_about = st.tabs([
    "🔍  Detect", "📊  Model Evaluation", "📁  Dataset & Training"
])


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DETECT
# ══════════════════════════════════════════════════════════════
with page_detect:
    st.write("Choose an input method, then click **Analyse Article** to see the model's prediction and confidence scores.")

    tab_text, tab_file = st.tabs(["✏️  Text / Headline", "📂  Upload File"])

    file_text     = ""
    uploaded_file = None

    with tab_text:
        st.markdown('<p class="section-label">News Article or Headline</p>',
                    unsafe_allow_html=True)
        user_input = st.text_area(
            label="text_input", label_visibility="collapsed",
            placeholder="Paste the full article or just a headline here…",
            height=200, key="user_text",
        )

    with tab_file:
        st.markdown('<p class="section-label">Upload .txt or .csv</p>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a news article", type=["txt", "csv"],
            label_visibility="collapsed",
            key=st.session_state.file_uploader_key,
        )
        if uploaded_file is not None:
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            if ext == "txt":
                file_text = uploaded_file.read().decode("utf-8")
                st.text_area("File contents", file_text, height=180, disabled=True)
            elif ext == "csv":
                df_up    = pd.read_csv(uploaded_file)
                st.dataframe(df_up.head(), use_container_width=True)
                text_col  = st.selectbox("Select the text column", df_up.columns)
                file_text = " ".join(df_up[text_col].astype(str).tolist())

    col_analyse, col_clear = st.columns([3, 1])
    with col_analyse:
        analyse_clicked = st.button("🔍  Analyse Article")
    with col_clear:
        def clear_form():
            st.session_state.user_text = ""
            st.session_state.file_uploader_key += 1
        st.button("🗑  Clear Inputs", on_click=clear_form)

    if analyse_clicked:
        has_text = user_input.strip() != ""
        has_file = uploaded_file is not None and file_text.strip() != ""

        if has_text and has_file:
            st.error("⚠️ **Conflict:** Use either the text box OR the file upload — not both.")
            st.info("Click **Clear Inputs** to reset, then choose one input method.")
        elif not has_text and not has_file:
            st.warning("Please enter some text or upload a file before analysing.")
        else:
            input_text = file_text if has_file else user_input
            source_tag = "📂 Uploaded File" if has_file else "✏️ Text Input"

            with st.spinner("Running Bidirectional LSTM inference…"):
                result = predict(input_text)

            if result is None:
                st.error("The input produced no usable words after preprocessing. "
                         "Try a longer article.")
            else:
                st.divider()
                label, real_prob, fake_prob = result['label'], result['real_prob'], result['fake_prob']

                if label == 'REAL':
                    css_cls = 'verdict-real'
                    icon, title = '✅', 'REAL NEWS DETECTED'
                    sub = f"The model is {real_prob:.1f}% confident this article is authentic."
                elif label == 'FAKE':
                    css_cls = 'verdict-fake'
                    icon, title = '🚨', 'FAKE NEWS DETECTED'
                    sub = f"The model is {fake_prob:.1f}% confident this is misinformation."
                else:
                    css_cls = 'verdict-uncertain'
                    icon, title = '⚠️', 'UNCERTAIN — MIXED SIGNALS'
                    sub = "The model cannot confidently classify this article. Verify independently."

                st.markdown(f"""
                <div class="{css_cls}">
                    <div class="verdict-title">{icon} {title}</div>
                    <div class="verdict-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                with c1:
                    st.markdown(f"""<div class="stat-box">
                        <div class="stat-value" style="color:#52b788">{real_prob:.1f}%</div>
                        <div class="stat-label">Real probability</div></div>""",
                        unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="stat-box">
                        <div class="stat-value" style="color:#e63946">{fake_prob:.1f}%</div>
                        <div class="stat-label">Fake probability</div></div>""",
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="stat-box">
                        <div class="stat-value" style="color:#4a4a7a">{result['word_count']}</div>
                        <div class="stat-label">Words analysed</div></div>""",
                        unsafe_allow_html=True)
                with c4:
                    st.plotly_chart(make_gauge(real_prob), use_container_width=True)

                fig_bar = go.Figure(go.Bar(
                    x=[real_prob, fake_prob], y=['Real', 'Fake'],
                    orientation='h', marker_color=['#52b788', '#e63946'],
                    text=[f"{real_prob:.1f}%", f"{fake_prob:.1f}%"],
                    textposition='inside', textfont=dict(color='white', family='Syne', size=13),
                ))
                fig_bar.update_layout(
                    height=120, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(range=[0,100], showgrid=False,
                               showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False), showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown(f"""<div class="info-box">
                    <b>Source:</b> {source_tag} &nbsp;·&nbsp;
                    <b>MAXLEN:</b> {MAXLEN} &nbsp;·&nbsp;
                    <b>Model:</b> Bidirectional LSTM
                </div>""", unsafe_allow_html=True)

                with st.expander("🔎 View preprocessed text"):
                    st.code(result['cleaned'][:1000] +
                            ("…" if len(result['cleaned']) > 1000 else ""),
                            language=None)

                if label == 'UNCERTAIN':
                    st.info("💡 **Tip:** Uncertain results often arise from short text, "
                            "satire, or opinion pieces. Cross-check with a trusted "
                            "fact-checking source.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
with page_eval:

    if eval_data is None:
        st.warning("""
**`eval_metrics.json` not found.**

Run the training notebook through to **Cell 26** (Save everything) to generate this file,
then place it in the same folder as `app.py`.
        """)
    else:
        e = eval_data

        # ── Section header ────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <p class="section-label">Model Performance on Held-Out Test Set</p>
            <p style="color:#666;font-size:0.95rem;margin:0;">
            All metrics computed on the 20 % test split that the model never saw during training.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Top scorecard row ──────────────────────────────────
        def badge(val, good=0.95, warn=0.85):
            cls = 'score-badge-good' if val >= good else \
                  ('score-badge-warn' if val >= warn else 'score-badge-danger')
            return cls

        sc1, sc2, sc3, sc4 = st.columns(4)

        with sc1:
            cls = badge(e['roc_auc'], 0.97, 0.90)
            st.markdown(f"""<div class="score-card">
                <div class="score-card-title">AUC-ROC</div>
                <div class="score-card-value {cls}">{e['roc_auc']:.4f}</div>
                <div class="score-card-sub">Threshold-independent · 1.0 = perfect</div>
            </div>""", unsafe_allow_html=True)

        with sc2:
            cls = badge(e['average_precision'], 0.97, 0.90)
            st.markdown(f"""<div class="score-card">
                <div class="score-card-title">Average Precision</div>
                <div class="score-card-value {cls}">{e['average_precision']:.4f}</div>
                <div class="score-card-sub">Area under PR curve</div>
            </div>""", unsafe_allow_html=True)

        with sc3:
            cls = badge(e['accuracy_at_0.5'], 0.97, 0.90)
            st.markdown(f"""<div class="score-card">
                <div class="score-card-title">Accuracy  (t = 0.50)</div>
                <div class="score-card-value {cls}">{e['accuracy_at_0.5']:.4f}</div>
                <div class="score-card-sub">Overall correct predictions</div>
            </div>""", unsafe_allow_html=True)

        with sc4:
            cls = badge(e['macro_f1_at_0.5'], 0.97, 0.90)
            st.markdown(f"""<div class="score-card">
                <div class="score-card-title">Macro F1  (t = 0.50)</div>
                <div class="score-card-value {cls}">{e['macro_f1_at_0.5']:.4f}</div>
                <div class="score-card-sub">Balanced across both classes</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROC + PR side by side ──────────────────────────────
        st.markdown('<p class="section-label">Curves</p>', unsafe_allow_html=True)
        col_roc, col_pr = st.columns(2)
        with col_roc:
            st.markdown("""<div class="insight-box">
                📌 <b>ROC Curve:</b> Plots True Positive Rate vs False Positive Rate across all thresholds.
                The closer the curve hugs the top-left corner, the better. AUC = 1.0 is perfect.
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_roc_chart(e), use_container_width=True)
        with col_pr:
            st.markdown("""<div class="insight-box">
                📌 <b>Precision-Recall Curve:</b> More informative than ROC for imbalanced datasets.
                High precision means few false alarms; high recall means few missed fakes.
                AP = area under this curve.
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_pr_chart(e), use_container_width=True)

        st.divider()

        # ── Threshold Sensitivity ──────────────────────────────
        st.markdown('<p class="section-label">Threshold Sensitivity</p>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="insight-box">
            🎚️ <b>How to use this:</b> Move the slider to explore how each metric
            changes as the decision threshold shifts. The <b>default threshold is 0.5</b>
            (probability > 0.5 → Real). Lowering it catches more fake news
            (higher Recall) but increases false alarms (lower Precision).
            The dashed green line marks the threshold that maximises F1.
        </div>""", unsafe_allow_html=True)

        selected_threshold = st.slider(
            "Decision Threshold",
            min_value=0.01, max_value=0.99, value=0.50, step=0.01,
            format="%.2f",
        )

        # Compute live metrics at selected threshold
        tx   = e['thresh_x']
        # Find nearest index in the sampled threshold array
        nearest = min(range(len(tx)), key=lambda i: abs(tx[i] - selected_threshold))

        live_acc  = e['thresh_acc'][nearest]
        live_prec = e['thresh_prec'][nearest]
        live_rec  = e['thresh_rec'][nearest]
        live_f1   = e['thresh_f1'][nearest]

        lc1, lc2, lc3, lc4 = st.columns(4)
        for col, label_t, val, color in [
            (lc1, 'Accuracy',  live_acc,  '#4a90d9'),
            (lc2, 'Precision', live_prec, '#f4a261'),
            (lc3, 'Recall',    live_rec,  '#e63946'),
            (lc4, 'F1-Score',  live_f1,   '#52b788'),
        ]:
            with col:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-value" style="color:{color}">{val:.3f}</div>
                    <div class="stat-label">{label_t} @ {selected_threshold:.2f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(make_threshold_chart(e, selected_threshold),
                        use_container_width=True)

        st.divider()

        # ── Confusion Matrix + Per-class bar ──────────────────
        st.markdown('<p class="section-label">Classification Breakdown</p>',
                    unsafe_allow_html=True)

        col_cm, col_bar = st.columns(2)
        with col_cm:
            st.markdown("""<div class="insight-box">
                📌 <b>Confusion Matrix</b> at the default threshold (0.5).
                <b>TP / TN</b> (diagonal) = correct predictions.
                <b>FP</b> = real news wrongly flagged as fake.
                <b>FN</b> = fake news that slipped through as real.
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_cm_chart(e), use_container_width=True)

            tp, tn = e['tp'], e['tn']
            fp, fn = e['fp'], e['fn']
            total  = tp + tn + fp + fn
            st.markdown(f"""
            <table class="cm-table">
              <tr>
                <td></td>
                <td><b>Pred: Fake</b></td>
                <td><b>Pred: Real</b></td>
              </tr>
              <tr>
                <td><b>Actual: Fake</b></td>
                <td class="cm-tn">{tn}<br><small>TN ({tn/total*100:.1f}%)</small></td>
                <td class="cm-fp">{fp}<br><small>FP ({fp/total*100:.1f}%)</small></td>
              </tr>
              <tr>
                <td><b>Actual: Real</b></td>
                <td class="cm-fn">{fn}<br><small>FN ({fn/total*100:.1f}%)</small></td>
                <td class="cm-tp">{tp}<br><small>TP ({tp/total*100:.1f}%)</small></td>
              </tr>
            </table>
            """, unsafe_allow_html=True)

        with col_bar:
            st.markdown("""<div class="insight-box">
                📌 <b>Per-Class Metrics</b> — comparing Fake vs Real class performance.
                For fake news detection, prioritise the <b>Fake class Recall</b> (catching all fakes)
                while keeping Precision high enough to avoid false alarms.
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(make_class_bar(e), use_container_width=True)

            # Detailed table
            metrics_df = pd.DataFrame({
                'Metric':    ['Precision', 'Recall', 'F1-Score'],
                'Fake (0)':  [f"{e['precision_fake_at_0.5']:.4f}",
                              f"{e['recall_fake_at_0.5']:.4f}",
                              f"{e['f1_fake_at_0.5']:.4f}"],
                'Real (1)':  [f"{e['precision_real_at_0.5']:.4f}",
                              f"{e['recall_real_at_0.5']:.4f}",
                              f"{e['f1_real_at_0.5']:.4f}"],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Optimal threshold recommendation ──────────────────
        opt_t = e['optimal_threshold_f1']
        st.markdown(f"""
        <div class="insight-box" style="border-color:#52b788; background:#f0fff6;">
            ✅ <b>Recommendation:</b> The threshold that maximises F1 on the test set is
            <b>{opt_t:.2f}</b> (accuracy = {e['accuracy_at_optimal']:.4f},
            Fake F1 = {e['f1_fake_at_optimal']:.4f},
            Real F1 = {e['f1_real_at_optimal']:.4f}).
            You can use the slider above to explore the trade-off and pick the best threshold
            for your use-case before updating <code>MAXLEN</code> and the prediction function.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — DATASET & TRAINING IMAGES
# ══════════════════════════════════════════════════════════════
with page_about:
    st.markdown('<p class="section-label">Dataset & Training Insights</p>',
                unsafe_allow_html=True)
    st.markdown("The model was trained on the Kaggle Fake and True News Dataset, which contains around 44,000 news articles labelled as REAL or FAKE." \
    "The dataset is balanced and includes a mix of headlines and full articles from various sources.<br><br>"\
    "The training process involved tokenising the text, padding sequences to a maximum length of 531 tokens, and feeding them into a Bidirectional LSTM architecture.<br><br>"\
    "Key training techniques included EarlyStopping to prevent overfitting and ReduceLROnPlateAU to fine-tune the learning rate.<br><br>"\
    "Below are some visualisations that provide insights into the dataset and the model's performance during training.",
                unsafe_allow_html=True)
    st.divider()

    images = [
        ("Subject_sample.png",   "Subject Distribution"),
        ("Confusion_matrix.png", "Confusion Matrix"),
        ("Class_report.png",     "Classification Report"),
        ("Training_Validation_Loss.png",      "Training vs Validation Loss Curves"),
        ("Training_Validation_Accuracy.png",  "Training vs Validation Accuracy Curves"),
        ("newplot.png",          "Distribution of sequence length on test data"),
    ]

    img_cols = st.columns(3)
    img_cols += st.columns(3)  # Add extra columns for the last two images
    for col, (fname, caption) in zip(img_cols, images):
        with col:
            if os.path.exists(fname):
                st.image(fname, caption=caption, use_container_width=True)
            else:
                st.markdown(
                    f'<div class="stat-box" style="padding:2rem;color:#aaa;font-size:0.8rem">'
                    f'{caption}<br><span style="font-size:0.7rem">({fname} not found)</span>'
                    f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div class="insight-box">
        <b>Model Architecture:</b> Bidirectional LSTM with two recurrent layers (128 + 64 units),
        SpatialDropout1D regularisation, a 128-dim Embedding layer, and a sigmoid output.
        Trained with <b>EarlyStopping</b> and <b>ReduceLROnPlateau</b> callbacks on an
        80/20 stratified split of the Kaggle Fake News Dataset (~44 000 articles).
    </div>
    """, unsafe_allow_html=True)