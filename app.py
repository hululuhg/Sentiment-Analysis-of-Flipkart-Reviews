import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load files
MODEL_PATH = "sentiment_model (1).h5"
TOKENIZER_PATH = "tokenizer (1).pkl"
LABEL_ENCODER_PATH = "label_encoder (1).pkl"
MAX_LEN = 100

@st.cache_resource
def load_files():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Load resources
model, tokenizer, label_encoder = load_files()

# UI
st.title("Sentiment Analysis App")
st.markdown("### Enter a sentence to analyze sentiment:")

user_input = st.text_input("Your input:")

if st.button("Analyze") and user_input.strip():
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    prediction = model.predict(padded)[0][0]
    label = "Positive" if prediction >= 0.5 else "Negative"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"**Prediction:** `{label}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
