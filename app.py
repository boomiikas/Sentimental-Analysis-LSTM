import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
# Clear the Streamlit cache to force the model to be re-loaded
st.cache_resource.clear()


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------------
# Custom Attention Layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh')

    def call(self, inputs):
        score = self.score_dense(inputs)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        # Only return the base config; Keras will handle sub-layers automatically
        base_config = super().get_config()
        return base_config

# -------------------------
# Load model, tokenizer, label encoder
@st.cache_resource
def load_sentiment_model():
    return load_model(
        "best_model.keras",
        compile=False,
        custom_objects={"AttentionLayer": AttentionLayer}    )


model = load_sentiment_model()


with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pkl", "rb") as handle:
    encoder = pickle.load(handle)

MAX_LEN = 50  # must match training

# -------------------------
# Streamlit page setup
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.write("Enter a sentence below and the model will predict its sentiment (Negative, Neutral, Positive).")

# -------------------------
# Input
user_input = st.text_area("Type your sentence here...", height=100)

# -------------------------
# Prediction
if st.button("🚀 Analyze Sentiment") and user_input.strip() != "":
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    
    # Predict
    pred = model.predict(padded)[0]
    idx = np.argmax(pred)
    sentiment = encoder.classes_[idx]
    confidence = float(pred[idx])

    st.success(f"🎯 Predicted Sentiment: **{sentiment.capitalize()}** (Confidence: {confidence:.2f})")

    # Probability bar chart
    prob_df = pd.DataFrame({
        "Sentiment": encoder.classes_,
        "Confidence": pred
    })

    fig, ax = plt.subplots()
    colors = ['red' if s=="negative" else 'gray' if s=="neutral" else 'green' for s in encoder.classes_]
    ax.bar(prob_df['Sentiment'], prob_df['Confidence'], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    for i, v in enumerate(prob_df['Confidence']):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig)

# -------------------------
# Sidebar
st.sidebar.header("About")
st.sidebar.write("""
This app uses a Bi-LSTM with Attention mechanism to classify text into:
- Negative
- Neutral
- Positive

Built with TensorFlow, Keras, and Streamlit. 🎉
""")