import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

app = FastAPI(title="E-Mariage Biometric Sovereignty Service")

# Configuration des CORS pour autoriser l'app React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def decode_image_base64(base64_str: str) -> np.ndarray:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("CV2 failed to decode image")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image Base64 invalide : {str(e)}")

@app.post("/compare")
async def compare_faces(
    image1: str = Form(..., description="Image 1 (CNI) en Base64"),
    image2: str = Form(..., description="Image 2 (Selfie en direct) en Base64"),
    detector_backend: str = Form("retinaface", description="Détecteur de visages"),
    model_name: str = Form("ArcFace", description="Modèle de comparaison")
):
    try:
        img1 = decode_image_base64(image1)
        img2 = decode_image_base64(image2)
        
        # ----------------- ÉTAPE 1 : CONTRÔLE DE VIVACITÉ (LIVENESS) -----------------
        # Nous extrayons le visage du selfie avec le contrôle anti-spoofing actif.
        try:
            selfie_faces = DeepFace.extract_faces(
                img_path=img2,
                detector_backend=detector_backend,
                anti_spoofing=True,
                enforce_detection=True
            )
        except Exception as e:
            return {
                "valide": False,
                "liveness": False,
                "score": 0,
                "decision": "REJETER",
                "message": "Aucun visage détecté sur le selfie. Veuillez bien vous cadrer face à la caméra."
            }

        # Vérification si le visage extrait est réel
        is_real = True
        for face_info in selfie_faces:
            # Si un des visages détectés est identifié comme un faux (photo d'une photo, écran, etc.)
            if not face_info.get("is_real", True):
                is_real = False
                break

        if not is_real:
            return {
                "valide": False,
                "liveness": False,
                "score": 0,
                "decision": "REJETER",
                "message": "⚠️ Alerte de sécurité : Détection de vivacité échouée (Présentation d'une photo ou d'un écran détectée)."
            }

        # ----------------- ÉTAPE 2 : COMPARAISON (MATCHING) -----------------
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        
        distance = result.get("distance", 1.0)
        verified = bool(result.get("verified", False))
        
        # Conversion de la distance de similarité (Cosine) en score sur 100
        similarity = max(0, min(100, (1 - distance) * 100))
        
        return {
            "valide": verified,
            "liveness": True,
            "score": round(similarity, 2),
            "decision": "VALIDER" if verified else "REJETER",
            "message": "Identité et vivacité confirmées ✅" if verified else "Le selfie ne correspond pas à la pièce d'identité."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de l'analyse biométrique : {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy"}
