import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import time

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="GRU Word Predictor - AI Language Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 1rem;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in-out;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeInUp 1.2s ease-in-out;
    }

    /* Animation Keyframes */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }

    /* Card Styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        color: white;
        animation: fadeInUp 0.8s ease-in-out;
    }

    .result-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 3px solid #667eea;
        animation: fadeInUp 1s ease-in-out;
    }

    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
    }

    /* Feature Box */
    .feature-box {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }

    .feature-box:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    /* Stats Box */
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        margin: 10px 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }

    .stats-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
    }

    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }

    .stats-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 5px;
        background: linear-gradient(120deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #667eea;
        padding: 15px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(120deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        animation: pulse 1s infinite;
    }

    /* Slider Styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    section[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600;
    }

    /* Divider */
    .custom-divider {
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        margin: 40px 0;
    }

    /* Word Display */
    .word-display {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 30px;
        display: inline-block;
        margin: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
        animation: fadeInUp 0.5s ease-in-out;
    }

    /* Example Text */
    .example-text {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-style: italic;
        color: #495057;
    }

    /* Loading Animation */
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: rotate 1s linear infinite;
        margin: 20px auto;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------
# Load Model Function
# -------------------------------
@st.cache_resource
def load_my_model():
    try:
        model = load_model("gru_word_prediction_model.h5")
        return model, True
    except Exception as e:
        return None, False


model, model_loaded = load_my_model()


# -------------------------------
# Load or Create Tokenizer
# -------------------------------
@st.cache_resource
def initialize_tokenizer():
    try:
        # Try to load saved tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer, True
    except:
        # Create tokenizer with sample data
        tokenizer = Tokenizer()
        sample_text = [
            "deep learning is a subset of machine learning",
            "machine learning is part of artificial intelligence",
            "gru is a recurrent neural network",
            "natural language processing uses deep learning",
            "artificial intelligence powers modern technology",
            "neural networks learn from data patterns",
            "recurrent networks process sequential data",
            "language models predict next words",
            "text generation requires large datasets",
            "word embeddings capture semantic meaning"
        ]
        tokenizer.fit_on_texts(sample_text)
        return tokenizer, False


tokenizer, tokenizer_loaded = initialize_tokenizer()
total_words = len(tokenizer.word_index) + 1
max_seq_len = 10


# -------------------------------
# Prediction Function
# -------------------------------
def predict_next_word(seed_text, next_words=5):
    predicted_words = []
    current_text = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

        if model_loaded:
            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        else:
            # Fallback random prediction if model not loaded
            predicted = np.random.randint(1, total_words)

        predicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                predicted_word = word
                break

        if predicted_word:
            current_text += ' ' + predicted_word
            predicted_words.append(predicted_word)
        else:
            break

    return current_text, predicted_words


# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 class='main-header'>🧠 GRU Word Prediction AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Next-Generation Language Model powered by Recurrent Neural Networks</p>",
            unsafe_allow_html=True)
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.markdown("---")

    # Model Status
    if model_loaded:
        st.markdown("""
            <div class='stats-box'>
                <div style='font-size: 2rem;'>✅</div>
                <div class='stats-label'>Model Loaded</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='info-card'>
                <div style='font-size: 1.5rem;'>⚠️</div>
                <div>Model file not found. Using demo mode.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Statistics
    st.markdown("### 📊 Model Statistics")

    st.markdown(f"""
        <div class='stats-box'>
            <div class='stats-number'>{total_words}</div>
            <div class='stats-label'>Vocabulary Size</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='stats-box'>
            <div class='stats-number'>{max_seq_len}</div>
            <div class='stats-label'>Max Sequence Length</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model Info
    st.markdown("### 🤖 About GRU")
    st.info("""
    **Gated Recurrent Units (GRU)** are a type of recurrent neural network that:

    - Process sequential data
    - Learn long-term dependencies
    - Predict next words in sequences
    - Power modern NLP applications
    """)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.success("""
    - Start with 2-3 words
    - Use complete sentences
    - Try different word counts
    - Experiment with styles
    """)

# -------------------------------
# Main Content
# -------------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("<h2 style='color: #667eea;'>📝 Input Your Text</h2>", unsafe_allow_html=True)

    # Text Input
    seed_text = st.text_input(
        "Enter your starting text:",
        value="deep learning",
        placeholder="Type your text here...",
        help="Enter 2-5 words to get started"
    )

    # Word Count Slider
    next_words = st.slider(
        "🎯 Number of words to predict:",
        min_value=1,
        max_value=10,
        value=5,
        help="Select how many words you want the model to generate"
    )

    # Predict Button
    predict_button = st.button("🚀 Generate Prediction", use_container_width=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='color: #667eea;'>📚 Examples</h2>", unsafe_allow_html=True)

    examples = [
        "deep learning",
        "machine learning is",
        "artificial intelligence",
        "neural networks",
        "natural language"
    ]

    st.markdown("**Try these examples:**")
    for example in examples:
        st.markdown(f"<div class='example-text'>• {example}</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction Results
# -------------------------------
if predict_button:
    if not seed_text.strip():
        st.error("❌ Please enter some text to generate predictions!")
    else:
        # Show loading animation
        with st.spinner('🤖 AI is generating predictions...'):
            time.sleep(0.5)  # Dramatic effect
            predicted_sentence, predicted_words = predict_next_word(seed_text, next_words)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # Results Section
        st.markdown("<h2 style='color: #667eea; text-align: center;'>✨ Prediction Results</h2>", unsafe_allow_html=True)

        col_result1, col_result2 = st.columns([2, 1], gap="large")

        with col_result1:
            # Full Predicted Text
            st.markdown("""
                <div class='result-card'>
                    <h3 style='color: #667eea; margin-bottom: 20px;'>📄 Complete Generated Text</h3>
                    <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 25px; border-radius: 15px; font-size: 1.3rem; 
                                line-height: 1.8; color: #333; font-weight: 500;
                                border: 2px solid #667eea;'>
            """ + predicted_sentence + """
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Predicted Words Breakdown
            st.markdown("""
                <div style='margin-top: 30px;'>
                    <h3 style='color: #667eea;'>🎯 Newly Generated Words</h3>
                </div>
            """, unsafe_allow_html=True)

            words_html = ""
            for i, word in enumerate(predicted_words, 1):
                words_html += f"<span class='word-display'>{i}. {word}</span>"

            st.markdown(f"<div style='text-align: center; margin: 20px 0;'>{words_html}</div>", unsafe_allow_html=True)

        with col_result2:
            # Statistics
            st.markdown("""
                <div class='prediction-card'>
                    <h4 style='margin-top: 0;'>📊 Generation Stats</h4>
                    <hr style='border-color: rgba(255,255,255,0.3);'>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <p style='margin: 15px 0;'><strong>Input Words:</strong> {len(seed_text.split())}</p>
                <p style='margin: 15px 0;'><strong>Generated Words:</strong> {len(predicted_words)}</p>
                <p style='margin: 15px 0;'><strong>Total Words:</strong> {len(predicted_sentence.split())}</p>
                <p style='margin: 15px 0;'><strong>Model Type:</strong> GRU</p>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Success Message
        st.success("✅ Prediction completed successfully!")

# -------------------------------
# Features Section
# -------------------------------
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #667eea; text-align: center;'>🌟 Key Features</h2>", unsafe_allow_html=True)

col_feat1, col_feat2, col_feat3 = st.columns(3)

with col_feat1:
    st.markdown("""
        <div class='feature-box'>
            <h3 style='color: #667eea;'>🎯 Smart Prediction</h3>
            <p>Advanced GRU architecture predicts contextually relevant words based on input sequence.</p>
        </div>
    """, unsafe_allow_html=True)

with col_feat2:
    st.markdown("""
        <div class='feature-box'>
            <h3 style='color: #667eea;'>⚡ Real-time</h3>
            <p>Lightning-fast predictions with optimized neural network inference.</p>
        </div>
    """, unsafe_allow_html=True)

with col_feat3:
    st.markdown("""
        <div class='feature-box'>
            <h3 style='color: #667eea;'>🔄 Flexible</h3>
            <p>Generate 1-10 words with customizable prediction length.</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='font-size: 0.9rem;'>Powered by TensorFlow & Keras | Built By Aqib Javed ❤️</p>
        <p style='font-size: 0.8rem;'>© 2024 GRU Word Prediction System. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)