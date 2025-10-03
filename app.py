from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
import easyocr
import numpy as np

app = FastAPI()

# Inicializa OCR reader apenas uma vez
reader = easyocr.Reader(['pt'])

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse({"error": "Arquivo precisa ser PDF"}, status_code=400)
    
    pdf_bytes = await file.read()

    # Converte PDF -> imagens
    images = convert_from_bytes(pdf_bytes)

    results = []
    for i, img in enumerate(images):
        img_np = np.array(img.convert("RGB"))
        text = reader.readtext(img_np, detail=0, paragraph=True)
        results.append({
            "page": i+1,
            "text": text
        })

    return {"ocr_result": results}
