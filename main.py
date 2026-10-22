import base64
import logging
import numpy as np
import cv2
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

logger = logging.getLogger("deepface-service")

app = FastAPI(title="E-Mariage Biometric Sovereignty Service")

# Le service n'est joignable que depuis le serveur (restriction d'adresses dans le proxy).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NO_FACE_MARKERS = ("Face could not be detected", "Exception while processing img")


@app.on_event("startup")
def load_models() -> None:
    # Chargement des modèles en mémoire au démarrage : sans cela le premier appel prend ~60 s,
    # au-delà du délai des Edge Functions.
    for detector in ("retinaface", "opencv", "ssd"):
        DeepFace.build_model(model_name=detector, task="face_detector")
    DeepFace.build_model(model_name="ArcFace", task="facial_recognition")
    DeepFace.build_model(model_name="Fasnet", task="spoofing")
    blank = np.full((224, 224, 3), 255, np.uint8)
    try:
        DeepFace.extract_faces(img_path=blank, detector_backend="retinaface", anti_spoofing=True, enforce_detection=False)
        DeepFace.verify(blank, blank, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)
    except Exception:
        logger.exception("Préchauffage incomplet")


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


def is_no_face_error(err: Exception) -> bool:
    return isinstance(err, ValueError) and any(m in str(err) for m in NO_FACE_MARKERS)


def rejected(message: str, liveness: bool = True) -> dict:
    return {"valide": False, "liveness": liveness, "score": 0, "decision": "REJETER", "message": message}


@app.post("/compare")
async def compare_faces(
    image1: str = Form(..., description="Image 1 (CNI) en Base64"),
    image2: str = Form(..., description="Image 2 (Selfie en direct) en Base64"),
    detector_backend: str = Form("retinaface", description="Détecteur de visages"),
    model_name: str = Form("ArcFace", description="Modèle de comparaison")
):
    img1 = decode_image_base64(image1)
    img2 = decode_image_base64(image2)

    # Une erreur technique (modèle absent, mémoire…) renvoie HTTP 500 : face-verify l'affiche comme
    # « service indisponible » et ne compte pas l'essai. Seule l'absence de visage est un refus.
    try:
        # ----------------- ÉTAPE 1 : CONTRÔLE DE VIVACITÉ (LIVENESS, FasNet) -----------------
        try:
            selfie_faces = DeepFace.extract_faces(
                img_path=img2,
                detector_backend=detector_backend,
                anti_spoofing=True,
                enforce_detection=True,
            )
        except ValueError as e:
            if is_no_face_error(e):
                return rejected("Aucun visage détecté sur le selfie. Veuillez bien vous cadrer face à la caméra.")
            raise

        # Le visage principal (le plus grand) doit être réel : photo d'une photo ou d'un écran refusée.
        main_face = max(selfie_faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        if not main_face.get("is_real", False):
            logger.info("Vivacité refusée (antispoof_score=%s)", main_face.get("antispoof_score"))
            return rejected(
                "⚠️ Contrôle de vivacité échoué : présentez votre visage en direct à la caméra "
                "(pas une photo ni un écran), avec un bon éclairage.",
                liveness=False,
            )

        # ----------------- ÉTAPE 2 : VISAGE SUR LA PIÈCE D'IDENTITÉ -----------------
        try:
            DeepFace.extract_faces(img_path=img1, detector_backend=detector_backend, enforce_detection=True)
        except ValueError as e:
            if is_no_face_error(e):
                return rejected("Aucun visage détecté sur la pièce d'identité. Reprenez la photo de la pièce, bien nette et sans reflet.")
            raise

        # ----------------- ÉTAPE 3 : COMPARAISON (MATCHING ArcFace) -----------------
        # Plusieurs visages sur la pièce (photo fantôme des CNI) : DeepFace retient la meilleure paire.
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Échec de l'analyse biométrique")
        raise HTTPException(status_code=500, detail=f"Échec de l'analyse biométrique : {str(e)}")

    distance = float(result.get("distance", 1.0))
    verified = bool(result.get("verified", False))
    # Conversion de la distance cosinus en score sur 100 (ArcFace valide sous 0,68, soit un score > 32).
    similarity = max(0.0, min(100.0, (1 - distance) * 100))

    return {
        "valide": verified,
        "liveness": True,
        "score": round(similarity, 2),
        "decision": "VALIDER" if verified else "REJETER",
        "message": "Identité et vivacité confirmées ✅" if verified else "Le selfie ne correspond pas à la pièce d'identité.",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
