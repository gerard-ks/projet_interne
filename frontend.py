import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.layers import Activation
import requests
import numpy as np

st.title("Authentification de la signature")

upload = st.file_uploader("Chargez l'image de votre signature", type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
  files = {"file": upload.getvalue()}

  try:
     # Réalisation de la requête API avec gestion des erreurs
    req = requests.post("http://localhost:8000/predict", files=files)
    req.raise_for_status()  # Lève une exception si la requête échoue
    resultat = req.json()
    rec = resultat["predictions"]
  
    # Dictionnaire de correspondance entre les classes et leurs descriptions
    class_descriptions = {
        0: "Signature de PersonA",
        1: "Signature de PersonB",
        2: "Signature de PersonC",
        3: "Signature de PersonD",
        4: "Signature de PersonE",
    }

    class_idx = np.argmax(rec)
    max_prob = rec[class_idx]

    if max_prob > 0.9:
        c1.image(Image.open(upload))
        # Afficher la description de la classe avec la probabilité maximale
        c2.write(f"Classe prédite: {class_descriptions[class_idx]}")
        c2.write(f"Probabilité: {max_prob:.2%}")
    else:
        c1.image(Image.open(upload))
        # Afficher les probabilités de toutes les classes
        for i, prob in enumerate(rec):
            c2.write(f"Probabilité: {prob:.2%}")
            c2.write(f"- {class_descriptions[i]}")

  except requests.exceptions.RequestException as e:
    st.error("Erreur de communication avec l'API: " + str(e))