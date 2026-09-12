import re
import string
import pickle

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEQUENCE_LENGTH = 9          # model was trained on windows of this length
INPUT_LENGTH = SEQUENCE_LENGTH - 1   # 8 tokens fed to the model
MODEL_PATH = "next_word_predictor.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 10%, #1b1035 0%, #0d0b1e 45%, #090714 100%);
    color: #eae7f5;
}

/* Hide default header/footer clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hero header */
.hero {
    padding: 2.1rem 2.4rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(124,58,237,0.28), rgba(56,189,248,0.14));
    border: 1px solid rgba(168,139,250,0.35);
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 40px rgba(88,28,235,0.25);
}
.hero h1 {
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #c4b5fd, #7dd3fc, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #b9b3d6;
    font-size: 1.02rem;
    margin-top: 0.4rem;
    max-width: 780px;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(168,139,250,0.25);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover {
    border-color: rgba(168,139,250,0.6);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #d8b4fe;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.78rem;
    color: #9d97bd;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* Section card */
.section-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(168,139,250,0.18);
    border-radius: 18px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e4defb;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Prediction chips */
.pred-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(56,189,248,0.2));
    border: 1px solid rgba(196,181,253,0.45);
    padding: 0.55rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.02rem;
    color: #f5f3ff;
    margin: 0.25rem;
}
.pred-chip .rank {
    font-size: 0.7rem;
    color: #d8b4fe;
    font-family: 'JetBrains Mono', monospace;
    opacity: 0.85;
}

/* Generated text box */
.gen-box {
    background: rgba(0,0,0,0.28);
    border: 1px solid rgba(168,139,250,0.3);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    font-size: 1.15rem;
    line-height: 1.9;
    color: #f1eefc;
}
.gen-box .seed { color: #a3a0b8; }
.gen-box .word-new {
    color: #7dd3fc;
    font-weight: 700;
    background: rgba(56,189,248,0.12);
    padding: 0 3px;
    border-radius: 4px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #38bdf8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.4);
}

textarea, .stTextArea textarea {
    background: rgba(0,0,0,0.25) !important;
    border-radius: 12px !important;
    color: #f1eefc !important;
    border: 1px solid rgba(168,139,250,0.3) !important;
    font-size: 1.05rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #150d2c, #0b0918);
    border-right: 1px solid rgba(168,139,250,0.15);
}

hr { border-color: rgba(168,139,250,0.15); }

.badge {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.4);
    color: #7dd3fc;
    border-radius: 8px;
    padding: 0.1rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# LOAD RESOURCES (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_resources()
vocab_size = min(tokenizer.num_words or len(tokenizer.word_index) + 1,
                  len(tokenizer.word_index) + 1)


# ----------------------------------------------------------------------------
# TEXT PROCESSING (mirrors training pipeline)
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_probabilities(seed_text: str, temperature: float) -> np.ndarray:
    cleaned = clean_text(seed_text)
    sequence = tokenizer.texts_to_sequences([cleaned])[0]
    sequence = [t for t in sequence if 1 < t < vocab_size]  # drop OOV / out-of-range
    sequence = sequence[-INPUT_LENGTH:]
    padded = pad_sequences([sequence], maxlen=INPUT_LENGTH, padding="pre")
    probs = model.predict(padded, verbose=0)[0]

    if temperature != 1.0:
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)
    return probs


def top_k_predictions(probs: np.ndarray, k: int):
    top_idx = np.argsort(probs)[-k:][::-1]
    results = []
    for idx in top_idx:
        word = tokenizer.index_word.get(idx, "")
        if word and word != "<OOV>":
            results.append((word, float(probs[idx])))
    return results


def sample_word(probs: np.ndarray) -> str:
    idx = np.random.choice(len(probs), p=probs)
    return tokenizer.index_word.get(idx, "")


def generate_text(seed_text: str, num_words: int, temperature: float, mode: str):
    generated = seed_text
    words_added = []
    for _ in range(num_words):
        probs = get_probabilities(generated, temperature)
        if mode == "Most likely (greedy)":
            idx = int(np.argmax(probs))
            next_word = tokenizer.index_word.get(idx, "")
        else:
            next_word = sample_word(probs)
        if not next_word:
            break
        generated += " " + next_word
        words_added.append(next_word)
    return generated, words_added


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="badge">BiLSTM · NEXT-WORD MODEL</span>
    <h1>🔮 Next Word Predictor</h1>
    <p>Type a phrase and watch a Bidirectional LSTM language model — trained on Arthur Conan Doyle's
    <i>The Adventures of Sherlock Holmes</i> — predict what comes next, one word or a whole sentence at a time.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MODEL INFO STRIP
# ----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, f"{vocab_size:,}", "Vocabulary size"),
    (c2, str(INPUT_LENGTH), "Input window (tokens)"),
    (c3, "128", "Embedding dim"),
    (c4, "256×2", "BiLSTM units"),
    (c5, f"{model.count_params():,}", "Total parameters"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Prediction Settings")

    num_words = st.slider(
        "Number of words to predict",
        min_value=1, max_value=30, value=5, step=1,
        help="How many additional words the model should generate after your text."
    )

    mode = st.radio(
        "Generation mode",
        ["Most likely (greedy)", "Sampled (creative)"],
        help="Greedy always picks the highest-probability word. Sampled introduces randomness for more varied text."
    )

    temperature = st.slider(
        "Temperature", min_value=0.2, max_value=1.5, value=0.7, step=0.05,
        help="Only used in Sampled mode. Lower = safer/more repetitive, higher = more creative/random.",
        disabled=(mode == "Most likely (greedy)")
    )

    top_k = st.slider(
        "Top-K candidates to display", min_value=3, max_value=15, value=6, step=1,
        help="How many alternative next-word candidates to show in the chart."
    )

    st.markdown("---")
    st.markdown("### 📖 About the model")
    st.markdown("""
    - **Architecture:** Embedding → Bidirectional LSTM(256) → Dense(256) → Softmax
    - **Training data:** *The Adventures of Sherlock Holmes* (Project Gutenberg)
    - **Objective:** predict the next token given the previous 8 tokens
    - **Loss:** sparse categorical cross-entropy
    """)

    st.markdown("---")
    if st.button("🎲 Try a random example"):
        st.session_state["seed_text"] = np.random.choice([
            "the king was",
            "it was a dark",
            "sherlock holmes said",
            "i have never seen",
            "the game is",
            "my dear watson",
            "there was no doubt",
        ])

# ----------------------------------------------------------------------------
# MAIN INPUT AREA
# ----------------------------------------------------------------------------
if "seed_text" not in st.session_state:
    st.session_state["seed_text"] = "the king was"

left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✍️ Your text</div>', unsafe_allow_html=True)
    seed_text = st.text_area(
        "Start typing…",
        value=st.session_state["seed_text"],
        height=120,
        label_visibility="collapsed",
        key="seed_input",
    )
    predict_btn = st.button("🔮 Predict next word(s)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn or seed_text.strip():
        if not seed_text.strip():
            st.warning("Please type some text first.")
        else:
            # single next-word prediction + top-k
            probs = get_probabilities(seed_text, temperature if mode != "Most likely (greedy)" else 1.0)
            candidates = top_k_predictions(probs, top_k)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Top next-word candidates</div>', unsafe_allow_html=True)

            if candidates:
                chip_html = ""
                for i, (word, p) in enumerate(candidates):
                    chip_html += f'<span class="pred-chip"><span class="rank">#{i+1}</span> {word} · {p*100:.1f}%</span>'
                st.markdown(chip_html, unsafe_allow_html=True)

                words, probs_list = zip(*candidates)
                fig = go.Figure(go.Bar(
                    x=[p * 100 for p in probs_list][::-1],
                    y=list(words)[::-1],
                    orientation="h",
                    marker=dict(
                        color=[p * 100 for p in probs_list][::-1],
                        colorscale=[[0, "#38bdf8"], [1, "#c4b5fd"]],
                    ),
                    text=[f"{p*100:.1f}%" for p in probs_list][::-1],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=30, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#eae7f5", family="Inter"),
                    xaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title=""),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No confident prediction found — try a longer or different phrase.")
            st.markdown('</div>', unsafe_allow_html=True)

            # multi-word generation
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">📝 Generated continuation (+{num_words} words)</div>', unsafe_allow_html=True)
            full_text, new_words = generate_text(seed_text, num_words, temperature, mode)
            highlighted = " ".join(
                [f'<span class="word-new">{w}</span>' for w in new_words]
            )
            st.markdown(
                f'<div class="gen-box"><span class="seed">{seed_text.strip()}</span> {highlighted}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Model architecture</div>', unsafe_allow_html=True)

    layers_info = []
    for layer in model.layers:
        try:
            shape = layer.output_shape
        except Exception:
            shape = "—"
        layers_info.append({
            "Layer": layer.__class__.__name__,
            "Output shape": str(shape),
            "Params": f"{layer.count_params():,}",
        })
    st.dataframe(layers_info, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Your text is lowercased, cleaned of punctuation/numbers/URLs (same as training).
    2. It's converted into token IDs using the fitted tokenizer, then truncated/padded
       to the model's fixed input window of **8 tokens**.
    3. The BiLSTM reads the sequence in both directions and outputs a probability
       distribution over the **5,000-word vocabulary**.
    4. In **greedy** mode the highest-probability word is picked; in **sampled**
       mode a word is drawn according to the (temperature-scaled) distribution —
       repeat for as many words as requested.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#65608a;font-size:0.85rem;margin-top:1rem;'>"
    "Built with Streamlit · TensorFlow/Keras · Model trained on <i>The Adventures of Sherlock Holmes</i>"
    "</p>", unsafe_allow_html=True
)
