import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Modell und Labels laden
@st.cache(allow_output_mutation=True)
def load_model_and_labels():
    model = load_model("keras_Model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model_and_labels()

st.set_page_config(page_title="T-Shirt Farberkennung", layout="centered")

st.title("T-Shirt Farberkennung für Farbenblinde")
st.write("Laden Sie ein Foto Ihres T-Shirts hoch. Die App erkennt, ob es rot, blau oder schwarz ist, und zeigt die Zuverlässigkeit in Prozent an.")

# Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen und anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bildverarbeitung
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)

    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Daten vorbereiten
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage durchführen
    prediction = model.predict(data)
    confidences = prediction[0]
    max_confidence = np.max(confidences)
    max_index = np.argmax(confidences)
    predicted_label = class_names[max_index]

    # Ergebnis interpretieren
    if max_confidence > 0.5:
        # Zuverlässigkeit in Prozent
        confidence_percent = round(max_confidence * 100, 2)
        if predicted_label.lower() == "rot":
            detected_text = f"Ein rotes T-Shirt wurde erkannt, mit einer Zuverlässigkeit von {confidence_percent} %."
        elif predicted_label.lower() == "blau":
            detected_text = f"Ein blaues T-Shirt wurde erkannt, mit einer Zuverlässigkeit von {confidence_percent} %."
        elif predicted_label.lower() == "schwarz":
            detected_text = f"Ein schwarzes T-Shirt wurde erkannt, mit einer Zuverlässigkeit von {confidence_percent} %."
        else:
            detected_text = "Die erkannte Farbe ist unklar. Bitte laden Sie ein anderes Foto hoch."
    else:
        detected_text = "Die Farbe des T-Shirts ist nicht eindeutig erkennbar. Bitte laden Sie ein anderes Foto hoch."

    st.success(detected_text)
else:
    st.info("Bitte laden Sie ein Bild hoch, um die Erkennung durchzuführen.")

# Hinweise für Barrierefreiheit
st.write("""
**Barrierefreiheitshinweis:**  
Diese Anwendung ist so gestaltet, dass sie für Menschen mit Farbenblindheit zugänglich ist.  
Die Ergebnisse werden klar in Textform angezeigt, und die App ist mit gut lesbaren Schriftarten gestaltet.  
Für noch bessere Zugänglichkeit empfehlen wir die Nutzung auf Geräten mit Sprachausgaben, falls vorhanden.
""")
