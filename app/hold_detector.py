import os
import cv2
import numpy as np
from ultralytics import YOLO

class HoldDetector:
    """détecteur de prises d'escalade utilisant un modèle yolo"""
    
    def __init__(self, model_path='PACTEv3.pt'):
        """initialise le détecteur avec le modèle spécifié"""
        # charger le modèle avec la classe YOLO d'ultralytics
        try:
            self.model = YOLO(model_path)
            self.conf = 0.25  # seuil de confiance
            self.iou = 0.45   # seuil iou
        except Exception as e:
            raise RuntimeError(f'erreur lors du chargement du modèle: {e}')
    
    def detect(self, image_path, return_image=False):
        """détecte les prises dans l'image et renvoie les boîtes"""
        # vérifier si le fichier existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'image non trouvée: {image_path}')
            
        # charger l'image pour l'affichage
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'impossible de lire l\'image: {image_path}')
        
        # convertir pour opencv
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        
        # faire les prédictions avec la classe YOLO
        results = self.model(image_path, conf=self.conf, iou=self.iou)
        
        # extraire les détections
        boxes = []
        
        for result in results:
            for box in result.boxes:
                # récupérer les coordonnées et la classe
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls.item())
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                boxes.append((cls, x1, y1, x2, y2))
        
        if return_image:
            return img_rgb, boxes, height, width
        return boxes

def load_image(image_path):
    """charge une image depuis un chemin"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'image non trouvée: {image_path}')
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'impossible de lire l\'image: {image_path}')
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def display_detections(img, boxes):
    """affiche l'image avec les détections"""
    import matplotlib.pyplot as plt
    
    # créer une copie
    img_copy = img.copy()
    
    # dessiner les boîtes
    for i, (class_id, x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # ajouter l'index
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.putText(img_copy, str(i), (center_x, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # afficher
    plt.figure(figsize=(10, 8))
    plt.imshow(img_copy)
    plt.title(f'{len(boxes)} prises détectées')
    plt.axis('off')
    plt.show()
    
    return img_copy
