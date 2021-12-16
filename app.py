# import numpy as np
import streamlit as st
# import tensorflow as tf
from PIL import Image
# from tensorflow.keras.preprocessing import image
# import numpy as np
import fastbook
from fastbook import *
path='Dataset'
dls= ImageDataLoaders.from_folder(path,train = "train",
                                   valid_pct=0.15,
                                   item_tfms=Resize(256),
                                   batch_tfms=None, bs = 8)
learn = cnn_learner(dls, models.resnet34)
learn.load('model')

#new
def predict_note_authentication(x):
    timg = TensorImage(image2tensor(x))
    tpil = PILImage.create(timg)
    result=learn.predict(tpil)
    return str(result[0])


def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Package detection </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        x = Image.open(uploaded_file)
        st.image(x, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = predict_note_authentication(x)
        st.write(label)

if __name__ == '__main__':
    main()