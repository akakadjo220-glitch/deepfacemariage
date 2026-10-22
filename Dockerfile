FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch (CPU seulement) : nécessaire au contrôle de vivacité FasNet de DeepFace.
RUN pip install --no-cache-dir torch==2.2.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement des poids pendant la construction (détecteurs, ArcFace, FasNet) : le premier appel
# n'a plus à les télécharger, et une dépendance manquante fait échouer la construction au lieu
# d'échouer en silence à chaque comparaison.
COPY warmup.py .
RUN python warmup.py && rm warmup.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
