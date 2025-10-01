import easyocr
from flask import Flask, jsonify
from PIL import Image

app = Flask(__name__)

IMAGE_PATH = 'testecpf.png'

print("Start initializing OCR reader")
reader = easyocr.Reader(['pt'])
print("Finished initializing OCR reader")

@app.route('/')
def read_aso():
    print("Start reading image")
    # Abre a imagem com PIL
    image_pil = Image.open(IMAGE_PATH)
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    result = reader.readtext(image_pil, detail=0)
    print("Finished reading image")

    print("Start creating response JSON")
    response = jsonify({
        'status': 'success',
        'data': result
    })
    print("Result: " + str(result))
    print("Finished creating response JSON")

    return response


if __name__ == '__main__':
    print("Starting Flask server")
    app.run(host='0.0.0.0', port=5000)
