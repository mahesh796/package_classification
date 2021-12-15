import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('weights-07-0.9828.h5',compile=False)

def predict_note_authentication(x):
    img = x.resize((256, 256), Image.ANTIALIAS)
    test_image = image.img_to_array(img)
    test_image = test_image / 255
    test_image = tf.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    prediction = np.argmax(prediction)
    classes=['carboard_boxes','Hessian_bags','Wooden_boxes']
    return classes[prediction]


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