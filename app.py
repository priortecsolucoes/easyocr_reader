import easyocr
from flask import Flask, jsonify

app = Flask(__name__)

# Caminho fixo da imagem do ASO
IMAGE_PATH = 'aso_image.jpg'

reader = easyocr.Reader(['pt'])  # Português


@app.route('/')
def read_aso():
    try:
        result = reader.readtext(IMAGE_PATH, detail=0)
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
