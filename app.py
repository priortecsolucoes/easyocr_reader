import easyocr
from flask import Flask, jsonify

app = Flask(__name__)

IMAGE_PATH = 'aso_image.png'

print("Start initializing OCR reader")
reader = easyocr.Reader(['pt'])
print("Finished initializing OCR reader")

@app.route('/')
def read_aso():
    print("Start reading image")
    result = reader.readtext(IMAGE_PATH, detail=0)
    print("Finished reading image")

    print("Start creating response JSON")
    response = jsonify({
        'status': 'success',
        'data': result
    })
    print("Result: " + str(response))
    print("Finished creating response JSON")

    return response


if __name__ == '__main__':
    print("Starting Flask server")
    app.run(host='0.0.0.0', port=5000)
