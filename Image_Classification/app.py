from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
model = load_model("model (2).hdf5")

# Replace this with actual class labels in correct order
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image).astype("float32") / 255.0

    # Optional: match CIFAR-10 training preprocessing
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std

    image = image.reshape(1, 32, 32, 3)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded."

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)

    predicted_label = class_labels[predicted_index]
    return render_template('result.html', predicted_class=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
