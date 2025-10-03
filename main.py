import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import easyocr

app = FastAPI()

# Inicializa OCR reader apenas uma vez
reader = easyocr.Reader(['pt'], recog_network='tiny', gpu=False)

@app.post("/upload-png/")
async def upload_png(file: UploadFile = File(...)):
    try:
        # Lê bytes do arquivo
        image_bytes = await file.read()
        # Tenta abrir com PIL e converter para RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Arquivo enviado não é uma imagem válida"}, status_code=400)

    # Converte para numpy array e faz OCR
    image_np = np.array(image)
    text = reader.readtext(image_np, detail=0, paragraph=True)

    return {"ocr_result": text}
