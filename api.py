from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import load_img
import numpy as np
import io
from PIL import Image

app = FastAPI()

def load():
    model_path = "MobileNetV3Large.h5"
    model = load_model(model_path, compile=False)
    return model

# chargement du modele
model = load()

# pretraitement
def preprocess(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img = load_img(img_bytes, target_size=(224, 224))
    # img = img.resize((224, 224))   redimensionner
    img = np.expand_dims(np.asarray(img), axis=0)  # Conversion et ajout de dimension
    img = preprocess_input(img)  # Normalisation
    return img 


@app.post("/predict")
async def predict(file: UploadFile):
    img_data = await file.read()

    # ouvir l'image
    img  = Image.open(io.BytesIO(img_data))

    # preprocessing
    img_processed = preprocess(img)

    # prediction
    predictions = model.predict(img_processed)
    probabilities = predictions[0].tolist()

    return {"predictions": probabilities}


