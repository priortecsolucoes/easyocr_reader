import io
import time
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import easyocr
from paddleocr import PaddleOCR

EXPECTED_PASSWORD = "Pr!ortecEasyOCR@2025"

app = Flask(__name__)

# -------------------------------
# 🔄 Inicialização dos modelos OCR
# -------------------------------
print("🔄 Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

print("🔄 Inicializando PaddleOCR (manuscritos na parte inferior)...")
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="pt",
    use_gpu=False
)
print("✅ PaddleOCR carregado com sucesso!")

# -------------------------------
# Função auxiliar para ler a parte inferior com PaddleOCR
# -------------------------------
def run_paddle_bottom_half(pil_image: Image.Image, percent_bottom: float = 0.35):
    width, height = pil_image.size
    crop_start = int(height * (1 - percent_bottom))
    bottom_crop = pil_image.crop((0, crop_start, width, height))
    image_np = np.array(bottom_crop)

    ocr_result = ocr.ocr(image_np)
    extracted_text = []
    for line in ocr_result:
        for word_info in line:
            extracted_text.append(word_info[1][0])
    return " ".join(extracted_text).strip()

# -------------------------------
# Endpoint principal
# -------------------------------
@app.route('/upload-png', methods=['POST'])
def upload_png():
    start = time.time()
    try:
        password = request.form.get('password', None)
        if password != EXPECTED_PASSWORD:
            return jsonify({'error': 'Senha inválida ou não fornecida.'}), 401
            
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

        file = request.files['file']

        keywords_str = request.form.get('keywords', None)
        percent_str = request.form.get('percent', '0.25')
        percent_paddle_str = request.form.get('percent_paddle', '0.35')

        keywords = []
        if keywords_str and keywords_str.strip():
            keywords = [k.strip().upper() for k in keywords_str.split(',') if k.strip()]

        try:
            percent = float(percent_str)
            percent_paddle = float(percent_paddle_str)
            if not (0 < percent <= 1) or not (0 < percent_paddle <= 1):
                raise ValueError
        except Exception:
            return jsonify({'error': 'Percentual inválido. Use um valor entre 0 e 1.'}), 400

        img = Image.open(file.stream).convert("RGB")
        width, height = img.size

        # -------------------------------
        # 🔹 Leitura com EasyOCR
        # -------------------------------
        if not keywords:
            image_np = np.array(img)
            easy_text_list = reader.readtext(image_np, detail=0, paragraph=True)
        else:
            top_crop = img.crop((0, 0, width, int(height * percent)))
            image_top_np = np.array(top_crop)
            partial_text = reader.readtext(image_top_np, detail=0, paragraph=True)
            partial_text_joined = " ".join(partial_text).upper()

            if any(keyword in partial_text_joined for keyword in keywords):
                bottom_crop = img.crop((0, int(height * percent), width, height))
                image_bottom_np = np.array(bottom_crop)
                rest_text = reader.readtext(image_bottom_np, detail=0, paragraph=True)
                easy_text_list = partial_text + rest_text
            else:
                total_time = time.time() - start
                return jsonify({
                    'status': 'not_identified',
                    'message': 'Documento não identificado pelas palavras-chave fornecidas no início.',
                    'ocr_result': partial_text,
                    'time': round(total_time, 2)
                })

        easy_text_joined = " ".join(easy_text_list)

        # -------------------------------
        # 🔹 Leitura da parte inferior com PaddleOCR
        # -------------------------------
        paddle_text = run_paddle_bottom_half(img, percent_bottom=percent_paddle)

        # -------------------------------
        # 🔹 Resultado combinado
        # -------------------------------
        combined_result = easy_text_joined.strip() + "\n---\n" + paddle_text.strip()
        total_time = time.time() - start

        return jsonify({
            'status': 'success',
            'easyocr_text': easy_text_joined,
            'paddle_text': paddle_text,
            'ocr_result': combined_result,
            'time': round(total_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "API OCR Flask pronta!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
