import re
import string
import pickle

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Class order — matches sklearn LabelEncoder alphabetical ordering, which is
# how the model's 13 softmax output indices were produced during training.
# ----------------------------------------------------------------------------
CLASSES = [
    "anger", "boredom", "empty", "enthusiasm", "fun", "happiness",
    "hate", "love", "neutral", "relief", "sadness", "surprise", "worry",
]

EMOTION_META = {
    "anger":      {"emoji": "😠", "color": "#FF4757"},
    "boredom":    {"emoji": "🥱", "color": "#8395A7"},
    "empty":      {"emoji": "🫥", "color": "#576574"},
    "enthusiasm": {"emoji": "🤩", "color": "#FFA502"},
    "fun":        {"emoji": "🎉", "color": "#FF6B9D"},
    "happiness":  {"emoji": "😄", "color": "#FFD93D"},
    "hate":       {"emoji": "🤬", "color": "#B33939"},
    "love":       {"emoji": "❤️", "color": "#FF4FA0"},
    "neutral":    {"emoji": "😐", "color": "#A4B0BE"},
    "relief":     {"emoji": "😌", "color": "#38ADA9"},
    "sadness":    {"emoji": "😢", "color": "#4A69BD"},
    "surprise":   {"emoji": "😲", "color": "#9C88FF"},
    "worry":      {"emoji": "😟", "color": "#F8A5C2"},
}

MAX_LEN = 100

# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nltk_resources():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model("emotion_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def clean_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def predict_emotion(text, model, tokenizer, stop_words, lemmatizer):
    cleaned = clean_text(text, stop_words, lemmatizer)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    return probs, cleaned


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1e2530 0%, #12151c 45%, #0a0c11 100%);
        color: #E8EAED;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 1.6rem 0 0.4rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B9D, #9C88FF, #38ADA9);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #8B93A6;
        font-size: 0.98rem;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #171B24 !important;
        border: 1px solid #2A3040 !important;
        border-radius: 14px !important;
        color: #E8EAED !important;
        font-size: 1.02rem !important;
        padding: 14px !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #9C88FF !important;
        box-shadow: 0 0 0 1px #9C88FF !important;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B9D 0%, #9C88FF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 0;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 18px rgba(156, 136, 255, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(156, 136, 255, 0.4);
    }

    .result-card {
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1.4rem 0 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-emoji { font-size: 3.6rem; margin-bottom: 0.3rem; }
    .result-label {
        font-size: 1.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-conf {
        color: #B8BFCC;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    .section-title {
        color: #C6CBD8;
        font-weight: 700;
        font-size: 1rem;
        margin: 1.4rem 0 0.4rem 2px;
        letter-spacing: 0.4px;
    }

    .example-chip {
        display: inline-block;
        background: #171B24;
        border: 1px solid #2A3040;
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.85rem;
        color: #9AA3B5;
        margin: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎭 Emotion Detector</h1>
        <p>Type a sentence and let the BiLSTM model read the emotion behind it</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model + tokenizer
# ----------------------------------------------------------------------------
with st.spinner("Loading model..."):
    stop_words, lemmatizer = load_nltk_resources()
    model, tokenizer = load_assets()

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

text = st.text_area(
    "Your sentence",
    value=st.session_state.text_input,
    placeholder="e.g. I just got the best news of my life!",
    height=120,
    label_visibility="collapsed",
)

predict_clicked = st.button("✨ Detect Emotion")

# ----------------------------------------------------------------------------
# Prediction + display
# ----------------------------------------------------------------------------
if predict_clicked:
    if not text.strip():
        st.warning("Please type a sentence first.")
    else:
        with st.spinner("Analyzing..."):
            probs, cleaned = predict_emotion(text, model, tokenizer, stop_words, lemmatizer)

        top_idx = int(np.argmax(probs))
        top_label = CLASSES[top_idx]
        top_conf = float(probs[top_idx])
        meta = EMOTION_META[top_label]

        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(160deg, {meta['color']}22, #12151c 70%); border-color: {meta['color']}55;">
                <div class="result-emoji">{meta['emoji']}</div>
                <div class="result-label" style="color: {meta['color']};">{top_label}</div>
                <div class="result-conf">Confidence: {top_conf*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sort all classes by probability for the chart
        order = np.argsort(probs)[::-1]
        sorted_labels = [CLASSES[i] for i in order]
        sorted_probs = [float(probs[i]) * 100 for i in order]
        sorted_colors = [EMOTION_META[l]["color"] for l in sorted_labels]

        st.markdown('<div class="section-title">📊 Probability breakdown</div>', unsafe_allow_html=True)

        fig = go.Figure(
            go.Bar(
                x=sorted_probs,
                y=[f"{EMOTION_META[l]['emoji']}  {l}" for l in sorted_labels],
                orientation="h",
                marker=dict(color=sorted_colors, line=dict(width=0)),
                text=[f"{p:.1f}%" for p in sorted_probs],
                textposition="outside",
                textfont=dict(color="#E8EAED", size=12),
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#C6CBD8", family="sans-serif"),
            margin=dict(l=10, r=40, t=10, b=10),
            height=460,
            xaxis=dict(
                title="Probability (%)",
                range=[0, max(sorted_probs) * 1.18],
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.06)",
            ),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 See cleaned text fed to the model"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.markdown(
    """
    <div style="text-align:center; color:#5C6379; font-size:0.8rem; margin-top:2.5rem;">
        Bidirectional LSTM · trained on 13 emotion classes
    </div>
    """,
    unsafe_allow_html=True,
)