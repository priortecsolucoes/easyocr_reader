import io
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import easyocr

app = FastAPI()

# Variável global para armazenar o leitor OCR
reader = None

@app.on_event("startup")
async def load_easyocr_model():
    """
    Carrega o modelo do EasyOCR na inicialização da API
    para evitar atrasos no primeiro request.
    """
    global reader
    print("🔄 [Startup] Carregando modelo EasyOCR (latin_g2)...")
    start_time = time.time()
    reader = easyocr.Reader(['pt'], recog_network='latin_g2', gpu=False)
    print(f"✅ [Startup] Modelo EasyOCR carregado em {time.time() - start_time:.2f}s")


@app.post("/upload-png/")
async def upload_png(file: UploadFile = File(...)):
    """
    Endpoint que processa uma imagem PNG/JPEG e retorna o texto lido via OCR.
    """
    global reader
    start_total = time.time()
    print("\n📥 [upload-png] Iniciando processamento...")

    if reader is None:
        print("❌ [upload-png] Modelo OCR não está carregado.")
        return JSONResponse({"error": "Modelo OCR ainda não foi carregado."}, status_code=503)

    try:
        start_read = time.time()
        image_bytes = await file.read()
        print(f"⏱️ [upload-png] Leitura do arquivo concluída em {time.time() - start_read:.2f}s")

        start_open = time.time()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"🖼️ [upload-png] Conversão para imagem concluída em {time.time() - start_open:.2f}s")

    except Exception as e:
        print(f"⚠️ [upload-png] Erro ao abrir imagem: {e}")
        return JSONResponse({"error": "Arquivo enviado não é uma imagem válida"}, status_code=400)

    start_np = time.time()
    image_np = np.array(image)
    print(f"🔢 [upload-png] Conversão para numpy array concluída em {time.time() - start_np:.2f}s")

    start_ocr = time.time()
    text = reader.readtext(image_np, detail=0, paragraph=True)
    print(f"🔍 [upload-png] OCR concluído em {time.time() - start_ocr:.2f}s")

    total_time = time.time() - start_total
    print(f"✅ [upload-png] Processo total concluído em {total_time:.2f}s")

    return {"ocr_result": text, "processing_time": total_time}
