FROM python:3.10-slim

# Remplacement de libgl1-mesa-glx par libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-téléchargement d'ArcFace, RetinaFace et des modèles de Liveness (Anti-Spoofing)
RUN python -c "from deepface import DeepFace; import numpy as np; img = np.zeros((100,100,3), dtype=np.uint8); \
    try: DeepFace.extract_faces(img_path=img, anti_spoofing=True, enforce_detection=False) except: pass; \
    try: DeepFace.verify(img, img, model_name='ArcFace', enforce_detection=False) except: pass"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
