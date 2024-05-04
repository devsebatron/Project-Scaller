from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Load your pre-trained model
model = load_model('/home/nevin/server_predict/model/trained_model.h5')

# Define class names and their corresponding treatments
class_names = {
    0: {'name': 'Actinic keratosis', 'treatment': 'Treatments may include cryotherapy, topical treatments, or photodynamic therapy.'},
    1: {'name': 'Atopic Dermatitis', 'treatment': 'Management includes moisturizers, topical steroids, and avoiding irritants.'},
    2: {'name': 'Benign keratosis', 'treatment': 'Treatment might involve cryotherapy, laser therapy, or surgical removal if necessary.'},
    3: {'name': 'Dermatofibroma', 'treatment': 'Often does not require treatment but can be removed surgically for cosmetic reasons or discomfort.'},
    4: {'name': 'Melanocytic nevus', 'treatment': 'Usually benign and requires no treatment, but monitoring for changes is important.'},
    5: {'name': 'Melanoma', 'treatment': 'Treatment options include surgical removal, immunotherapy, targeted therapy, chemotherapy, and radiation therapy.'},
    6: {'name': 'Squamous cell carcinoma', 'treatment': 'Treatment generally involves surgical removal, which may be followed by radiation or chemotherapy in advanced cases.'},
    7: {'name': 'Tinea Ringworm Candidiasis', 'treatment': 'Antifungal medications are the primary treatment, available in topical or oral form depending on the severity.'},
    8: {'name': 'Vascular lesion', 'treatment': 'Treatment options may include laser therapy, sclerotherapy, or surgery, depending on the type and location of the lesion.'}
}

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', message='No file uploaded')

        file = request.files['file']

        # Image preprocessing
        image = Image.open(file).convert('L')
        image = image.resize((28, 28))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        image = image / 255.0

        # Prediction
        predictions = model.predict(image)
        class_idx = np.argmax(predictions, axis=1)[0]

        # Retrieve class name and treatment from the class_names dictionary
        if class_idx in class_names:
            class_info = class_names[class_idx]
            return render_template('results.html', class_name=class_info['name'], treatment=class_info['treatment'])
        else:
            return render_template('error.html', message='Unknown class predicted')

if __name__ == '__main__':
    app.run(debug=True)
