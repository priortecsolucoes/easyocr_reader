import os
import cv2
import numpy as np
import torch
import easyocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

print("🔄 Inicializando EasyOCR...")
easyocr_reader = easyocr.Reader(['pt'])
print("✅ EasyOCR carregado com sucesso!")

print("🔄 Inicializando docTR (manuscritos na parte inferior)...")
doctr_model = ocr_predictor(pretrained=True)
print("✅ docTR carregado com sucesso!")

def read_text_easyocr(image_path):
    print("📘 Iniciando leitura com EasyOCR...")
    results = easyocr_reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results)

def read_text_doctr(image_path):
    print("✍️ Iniciando leitura manuscrita com docTR (parte inferior)...")

    # Carrega imagem
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Recorta a parte inferior (ex: 40% inferiores do documento)
    bottom_crop = img[int(h * 0.6):, :]

    # Salva crop temporário
    cropped_path = "bottom_crop.jpg"
    cv2.imwrite(cropped_path, bottom_crop)

    # Processa com docTR
    doc = DocumentFile.from_images(cropped_path)
    result = doctr_model(doc)

    # Concatena o texto detectado
    doctr_text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                doctr_text += " ".join([word.value for word in line.words]) + "\n"

    return doctr_text.strip()

def process_document(image_path):
    try:
        easy_text = read_text_easyocr(image_path)
        doctr_text = read_text_doctr(image_path)
        return {
            "easyocr_text": easy_text,
            "doctr_text": doctr_text
        }
    except Exception as e:
        print(f"❌ Erro na API OCR: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    test_image = "documento_teste.jpg"
    if os.path.exists(test_image):
        result = process_document(test_image)
        print("📄 Resultado EasyOCR:\n", result["easyocr_text"][:500])
        print("🖋️ Resultado docTR:\n", result["doctr_text"][:500])
    else:
        print("⚠️ Nenhum arquivo 'documento_teste.jpg' encontrado para teste.")
