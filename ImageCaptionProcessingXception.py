import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences


def extract_features(img_path, model):
    img_path = os.path.join(img_path)
    try:
        image = Image.open(img_path)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        mage = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def generate_caption(img_path, xception_model, model, tokenizer, max_length):
    photo = extract_features(img_path, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length) + '.'
    description = description.replace('start', '')
    description = description.replace('end', '')
    description = description.capitalize()
    return description
