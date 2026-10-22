# Précharge les modèles utilisés par main.py. Toute erreur fait échouer la construction de l'image.
from deepface import DeepFace

for detector in ("retinaface", "opencv", "ssd"):
    DeepFace.build_model(model_name=detector, task="face_detector")
DeepFace.build_model(model_name="ArcFace", task="facial_recognition")
DeepFace.build_model(model_name="Fasnet", task="spoofing")
print("Modèles DeepFace prêts : retinaface, opencv, ssd, ArcFace, Fasnet")
