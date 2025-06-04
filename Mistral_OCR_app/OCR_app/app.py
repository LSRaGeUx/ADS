import os
from flask import Flask, request, render_template, jsonify
import requests
from dotenv import load_dotenv
from mistralai import Mistral
import base64

load_dotenv()
app = Flask(__name__)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
model = "mistral-ocr-latest"
client = Mistral(api_key=MISTRAL_API_KEY)

@app.route('/')
def index():
    return render_template("index.html")

def encode_image(file):
    """Encode the uploaded file to base64."""
    try:
        return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding file: {e}")
        return None

@app.route('/ocr', methods=['POST'])
def ocr():
    image = request.files['image']
    if not image:
        return jsonify({'error': 'No image file provided'}), 400

    base64_img = encode_image(image)
    if not base64_img:
        return jsonify({'error': 'Failed to encode image'}), 500

    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_img}" 
            },
            include_image_base64=True
        )
        # Check if the response contains the expected data
        print(ocr_response)  # For debugging purposes
        if hasattr(ocr_response, 'pages'):  # Adjust based on the SDK's response structure
            text = ocr_response.pages[0].markdown or '[Aucune transcription]'
            return jsonify({'text': text})
        else:
            return jsonify({'error': 'Unexpected response format from Mistral API'}), 500
    except Exception as e:
        print(f"Error processing OCR: {e}")
        return jsonify({'error': 'Erreur Mistral API'}), 500

if __name__ == "__main__":
    app.run(debug=True)
