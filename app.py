import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load model & tokenizer
# -------------------------------
model = load_model('lstm_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Reverse word index
reverse_index = {v: k for k, v in tokenizer.word_index.items()}
reverse_index[0] = ''  # padding

# Max sequence length (use same as training)
max_len = 44

# -------------------------------
# Prediction function (improved)
# -------------------------------
def predict_next_word(seed_text, num_words=10):
    text = seed_text.lower().strip()

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]

        # If no known words → stop early
        if len(token_list) == 0:
            return text + " (No known words in vocab)"

        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        predicted = model.predict(token_list, verbose=0)[0]

        # 🔥 Better sampling instead of argmax
        predicted_index = np.random.choice(len(predicted), p=predicted)

        # Skip OOV token
        if predicted_index == 1:
            continue

        output_word = reverse_index.get(predicted_index, '')

        # Stop if no valid word
        if output_word == '':
            break

        text += " " + output_word

    return text


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title("🧠 LSTM Next Word Prediction")
st.write("⚡ Enter news-style text for best results")

# Input
seed_text = st.text_input("Enter seed text:", placeholder="e.g. the government")

# Slider
num_words = st.slider("Number of words to predict:", 1, 20, 10)

# Debug info (optional but useful)
with st.expander("🔍 Debug Info"):
    st.write("Vocabulary size:", len(tokenizer.word_index))
    st.write("Max sequence length:", max_len)

# Button
if st.button("Predict"):
    if seed_text.strip() != "":
        result = predict_next_word(seed_text, num_words)
        st.success(result)
    else:
        st.warning("⚠️ Please enter some text")
        