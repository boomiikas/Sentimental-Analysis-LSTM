import streamlit as st
import re
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

# -------------------------
# Define Custom Attention Layer
# -------------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh')

    def build(self, input_shape):
        # Build Dense layer manually to avoid H5 loading errors
        self.score_dense.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        score = self.score_dense(inputs)  # [batch, timesteps, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        return config

# -------------------------
# NLTK Setup (silent downloads for cloud)
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab',quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -------------------------
# Load Tokenizer & Label Encoder
# -------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------
# Load Trained Model (.h5)
# -------------------------
max_len = 50  # must match training
embedding_dim = 100  # replace with your actual embedding/input size

try:
    model = load_model("rnnmodel.h5", custom_objects={"AttentionLayer": AttentionLayer})
except Exception:
    # If loading fails on cloud, rebuild first
    model = load_model("rnnmodel.h5", custom_objects={"AttentionLayer": AttentionLayer})
    dummy_input = tf.zeros((1, max_len, embedding_dim))
    model.build(dummy_input.shape)
    model = load_model("rnnmodel.h5", custom_objects={"AttentionLayer": AttentionLayer})

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean text
        clean_text = clean_preprocess(user_input)

        # Convert to sequence
        seq = tokenizer.texts_to_sequences([clean_text])
        pad_seq = pad_sequences(seq, maxlen=max_len)

        # Predict
        pred = model.predict(pad_seq)
        pred_label = np.argmax(pred, axis=1)
        sentiment = le.inverse_transform(pred_label)[0]

        st.success(f"Predicted Sentiment: **{sentiment}**")
