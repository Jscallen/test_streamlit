import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from streamlit_drawable_canvas import st_canvas
import cv2

model = tf.keras.models.load_model('cnn_chiffres_arturo.h5')

dataset_path = 'fileforstreamlit.csv'
df = pd.read_csv(dataset_path)

def row_to_image(row):
    try:
        values = row[1:].values
        if len(values) != 784:
            raise ValueError(f"Expected 784 values, got {len(values)}")
        pixels = values.reshape((28, 28)).astype(np.float32)
        return pixels
    except Exception as e:
        st.error(f"Error in row_to_image: {e}")
        return None

def predict(image):
    image = np.expand_dims(image, axis=-1) 
    image = np.expand_dims(image, axis=0)  
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

pages = ["Page Principale", "Dessiner un Chiffre", "Page 3"]
page = st.sidebar.selectbox("Choisissez une page", pages)

if page == "Page Principale":
    st.title("Application de Prédiction d'Images")

    if st.button("Pick a Random Image"):
        random_row = df.sample(n=1).iloc[0]
        image = row_to_image(random_row)
        if image is not None:
            st.session_state['image'] = image
            st.session_state['prediction'] = None

    if 'image' in st.session_state:
        if isinstance(st.session_state['image'], np.ndarray) and st.session_state['image'].shape == (28, 28):
            try:
                st.image(st.session_state['image'], caption='Image sélectionnée', clamp=True, channels='GRAY')
            except Exception as e:
                st.error(f"Error displaying image: {e}")
        else:
            st.error("Invalid image format. The image must be a numpy array of shape (28, 28).")

        if st.button("Predict"):
            if 'image' in st.session_state:
                st.session_state['prediction'] = predict(st.session_state['image'])

    if 'prediction' in st.session_state and st.session_state['prediction'] is not None:
        st.write(f"La prédiction pour cette image est : {st.session_state['prediction']}")

elif page == "Dessiner un Chiffre":
    st.title("Dessiner un Chiffre")

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0  # Normaliser l'image entre 0 et 1
        img = np.expand_dims(img, axis=-1)

        if st.button("Predict Drawing"):
            prediction = predict(img)
            st.write(f"La prédiction pour ce dessin est : {prediction}")

elif page == "Page 3":
    st.title("Page 3")
    st.write("Contenu de la troisième page à ajouter ici.")
