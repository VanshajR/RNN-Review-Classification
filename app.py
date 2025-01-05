import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import re
import streamlit as st

# Load IMDB word index and model
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

custom_objects = {'Orthogonal': Orthogonal}
model = load_model('rnn_model_imdb.h5',custom_objects=custom_objects)

# Functions to decode the review and to pad the review
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_text(text):
    # Clean the input text
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = text.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(text):
    padded_review = preprocess_text(text)
    pred = model.predict(padded_review)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
    return sentiment, pred[0][0]

# Streamlit app
st.set_page_config(page_title="Review Sentiment Classifier", layout="centered",page_icon="🎬")

# App Title
st.title("🎥 Review Sentiment Classifier")
st.markdown(
    """
    Predict whether a movie review is **Positive** or **Negative** using a trained RNN model.
    """
)

# Input Section
st.header("Enter a Movie Review")
user_input = st.text_area(
    "Write your review below:", 
    placeholder="Type your movie review here...",
    height=150
)

# Button to classify the sentiment
if st.button("Classify Sentiment"):
    if user_input.strip():
        # Get prediction
        sentiment, confidence = predict_sentiment(user_input)
        confidence_percentage = confidence * 100

        # Display results
        st.subheader("Prediction Result")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** {confidence_percentage:.2f}%")
        
        # Display a message based on sentiment
        if sentiment == "Positive":
            st.success("🎉 This review is positive!")
        elif sentiment == "Negative":
            st.error("😞 This review is negative.")
        else:
            st.warning("⚠️ Please frame a proper sentence in English.")
    else:
        st.warning("⚠️ Please enter a review to classify.")

# Add a sidebar with additional information
st.sidebar.header("About This App")
st.sidebar.markdown(
    """
    - **Model Used:** Simple RNN trained on IMDB dataset.
    - **Input:** A movie review text.
    - **Output:** Predicted sentiment (Positive/Negative) with confidence.
    - Built using **Streamlit** and **TensorFlow**.
    """
)

st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1. Type or paste a movie review (in English) in the text box.
    2. Click the **Classify Sentiment** button to get the result.
    3. The app will display whether the review is **Positive** or **Negative** along with a confidence score.
    """
)
