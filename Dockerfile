FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Génération propre du script d'initialisation et exécution
RUN printf "from deepface import DeepFace\n\
import numpy as np\n\
img = np.zeros((100, 100, 3), dtype=np.uint8)\n\
try:\n\
    DeepFace.extract_faces(img_path=img, anti_spoofing=True, enforce_detection=False)\n\
except Exception:\n\
    pass\n\
try:\n\
    DeepFace.verify(img, img, model_name='ArcFace', enforce_detection=False)\n\
except Exception:\n\
    pass\n\
" > warmup.py && python warmup.py && rm warmup.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
