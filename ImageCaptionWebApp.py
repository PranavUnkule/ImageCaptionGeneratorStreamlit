import streamlit as st
from ImageCaptionProcessingXception import generate_caption
import tempfile
from pickle import load
import tensorflow as tf

def main():
    st.title('Image Caption Generator using CNN - LSTM Model')
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    st.sidebar.write("**Members**")
    st.sidebar.write("-- Pranav Unkule")
    st.sidebar.write("-- Aniket Singh")
    st.sidebar.write("-- Sanchit Agarkar")
    st.sidebar.write("-- Pratik Saurkar")

    image_file = st.file_uploader(
        "Upload Image", type=['jpg','jpeg','png'])

    # tokenizer
    max_length = 32
    tokenizer = load(open("tokenizer.p", "rb"))

    # load models
    model_dir = 'Models/model_lstm.json'
    model_weights_dir = 'Models/model_lstm_weights.hdf5'
    with open(model_dir, 'r') as json_file:
        json_saved_model = json_file.read()
    model = tf.keras.models.model_from_json(json_saved_model)
    model.load_weights(model_weights_dir)

    model_dir = 'Models/Xception.json'
    model_weights_dir = 'Models/Xception_Weights.hdf5'
    with open(model_dir, 'r') as json_file:
        json_saved_model = json_file.read()
    xception_model = tf.keras.models.model_from_json(json_saved_model)
    xception_model.load_weights(model_weights_dir)

    if image_file:
        st.header("Uploaded Image")
        FRAME_WINDOW = st.image([])
        FRAME_WINDOW.image(image_file)
        if st.button("Generate Caption", key="1"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(image_file.read())
            caption = generate_caption(tfile.name, xception_model, model, tokenizer, max_length)
            st.subheader("Caption: {}".format(caption))


if __name__ == "__main__":
    main()
