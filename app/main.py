import os
import sys
import argparse

from app.hold_detector import HoldDetector, load_image, display_detections
from app.path_finder import find_climbing_path

def main():
    """fonction principale du programme"""
    # définir le chemin par défaut vers le modèle
    default_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'models', 'PACTEv3.pt')
    
    # analyser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='détection de prises d\'escalade et planification de chemin')
    parser.add_argument('image_path', help='chemin vers l\'image de la paroi')
    parser.add_argument('--model', default=default_model_path, help='chemin vers le modèle yolo')
    parser.add_argument('--show-detections', action='store_true', help='afficher les détections brutes')
    parser.add_argument('--conf', type=float, default=0.25, help='seuil de confiance pour les détections')
    
    args = parser.parse_args()
    
    # vérifier si le fichier existe
    if not os.path.exists(args.image_path):
        print(f'erreur: l\'image {args.image_path} n\'existe pas')
        return 1
    
    # vérifier si le modèle existe
    if not os.path.exists(args.model):
        print(f'erreur: le modèle {args.model} n\'existe pas')
        return 1
    
    try:
        print('initialisation du détecteur de prises...')
        detector = HoldDetector(model_path=args.model)
        detector.conf = args.conf  # ajuster le seuil de confiance
        
        print(f'analyse de l\'image: {args.image_path}')
        img, boxes, _, _ = detector.detect(args.image_path, return_image=True)
        
        if not boxes:
            print('aucune prise détectée dans l\'image')
            return 1
            
        print(f'{len(boxes)} prises détectées')
        
        # afficher les détections si demandé
        if args.show_detections:
            display_detections(img, boxes)
        
        # paramètres par défaut pour le chemin
        params = {
            'eps': 118,
            'max_reach': 120,
            'max_limb_separation': 200,
            'foot_start': 0,
            'hand_start': 0,
            'finish': len(boxes) - 1
        }
        
        # lancer l'interface de recherche de chemin
        find_climbing_path(img, boxes, params)
        
        return 0
        
    except Exception as e:
        print(f'erreur: {str(e)}')
        return 1

if __name__ == '__main__':
    sys.exit(main())
