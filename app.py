import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


st.title("Clasificador de Reseñas de Películas 🎬")
st.write("Modelo entrenado en IMDb con Keras")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_imdb.h5")

model = load_model()
num_words = 10000
word_index = imdb.get_word_index()


index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"


def encode_text(text):
    words = text.lower().split()
    encoded = [1]  # <START>
    for word in words:
        idx = word_index.get(word, 2)  # <UNK> = 2
        if idx < num_words:
            encoded.append(idx)
    return pad_sequences([encoded], maxlen=200)


user_input = st.text_area("Escribe una reseña de película:", "")

if st.button("Clasificar"):
    if user_input.strip() == "":
        st.warning("Escribe algo primero.")
    else:
        encoded = encode_text(user_input)
        prediction = model.predict(encoded)[0][0]
        st.write("Probabilidad de reseña positiva:", f"{prediction:.2%}")
        if prediction > 0.5:
            st.success("👍 Reseña positiva")
        else:
            st.error("👎 Reseña negativa")
