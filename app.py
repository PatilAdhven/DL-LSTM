import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_models():
    model = load_model("lstm_text_predictor.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_models()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

max_sequence_len = 20

def predict(text):
    if not text:
        return ""
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return "No prediction (unknown words)"
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return "No prediction"

st.title("LSTM Next Word Predictor")
st.write("Enter a sequence of words and the model will predict the next word.")

input_text = st.text_input("Seed Text", placeholder="e.g., deep learning is")

if st.button("Predict"):
    if input_text:
        with st.spinner("Predicting..."):
            prediction = predict(input_text)
        st.success(f"Predicted next word: **{prediction}**")
    else:
        st.warning("Please enter some text.")