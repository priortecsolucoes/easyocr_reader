from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import easyocr
import numpy as np

app = FastAPI()

# Inicializa OCR reader apenas uma vez
reader = easyocr.Reader(['pt'])

@app.post("/upload-png/")
async def upload_png(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Arquivo precisa ser PNG ou JPEG"}, status_code=400)
    
    # Lê a imagem em memória
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"Não foi possível abrir a imagem: {str(e)}"}, status_code=400)

    # Converte para numpy array e faz OCR
    image_np = np.array(image)
    text = reader.readtext(image_np, detail=0, paragraph=True)

    return {"ocr_result": text}
