import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.cluster import DBSCAN
from random import randint
from matplotlib.widgets import Slider, Button, TextBox

def load_image_and_boxes(image_path):
    """charge l'image et les boîtes de prise"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    # chercher le fichier d'étiquettes correspondant
    labels_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    
    boxes = []
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])

                # format YOLO: x_center, y_center, w, h (normalisé)
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # conversion en coordonnées absolues
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)
                
                boxes.append((class_id, x1, y1, x2, y2))
                
    return img, boxes, height, width

def display_numbered_boxes(img, boxes):
    """affiche les boîtes numérotées pour permettre la sélection"""
    display_img = img.copy()
    
    # dessiner toutes les boîtes en rouge
    for i, (class_id, x1, y1, x2, y2) in enumerate(boxes):
        # dessiner la boîte
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # calculer le centre pour le texte
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # ajouter le numéro (index + 1 pour commencer à 1)
        cv2.putText(display_img, str(i), (center_x, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_img

def cluster_boxes(boxes, eps=118):
    """groupe les boîtes en clusters en utilisant dbscan"""
    # calculer les centres des boîtes
    box_centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for _, x1, y1, x2, y2 in boxes])
    
    # appliquer dbscan
    clustering = DBSCAN(eps=eps, min_samples=1).fit(box_centers)
    cluster_labels = clustering.labels_
    
    # génération de couleurs pour les clusters
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    cluster_colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(n_clusters)]
    
    return box_centers, cluster_labels, cluster_colors

def draw_clustered_boxes(img, boxes, cluster_labels, cluster_colors, foot_start, hand_start, finish):
    """dessine les boîtes avec les couleurs de cluster et marque les départs/arrivées"""
    cluster_img = img.copy()
    
    for i, (class_id, x1, y1, x2, y2) in enumerate(boxes):
        # déterminer la couleur du contour
        if i == foot_start or i == hand_start:
            border_color = (0, 255, 0)  # vert pour les départs
        elif i == finish:
            border_color = (0, 255, 0)  # vert pour l'arrivée
        else:
            border_color = (255, 0, 0)  # rouge pour les autres boîtes
        
        # dessiner le contour
        cv2.rectangle(cluster_img, (x1, y1), (x2, y2), border_color, 2)
        
        # remplir avec la couleur du cluster
        if cluster_labels[i] != -1:  # points non-bruit
            cluster_color = cluster_colors[cluster_labels[i]]
            # créer un calque avec transparence
            overlay = cluster_img.copy()
            cv2.rectangle(overlay, (x1+2, y1+2), (x2-2, y2-2), cluster_color, -1)
            alpha = 0.5  # facteur de transparence
            cluster_img = cv2.addWeighted(overlay, alpha, cluster_img, 1-alpha, 0)
    
    return cluster_img

def find_path(boxes, box_centers, cluster_labels, foot_start, hand_start, finish, 
              max_reach=120, max_limb_separation=200):
    """trouve le meilleur chemin en utilisant la recherche a*"""
    # calculer les distances entre toutes les prises
    dist_matrix = np.zeros((len(boxes), len(boxes)))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            dist_matrix[i, j] = np.linalg.norm(box_centers[i] - box_centers[j])

    # créer un graphe pour la recherche de chemin
    G = nx.Graph()
    for i in range(len(boxes)):
        G.add_node(i, pos=tuple(box_centers[i]))

    # ajouter des arêtes entre les prises atteignables
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if 0 < dist_matrix[i, j] <= max_reach:
                # ajuster le poids en fonction du cluster
                weight = dist_matrix[i, j]
                if cluster_labels[i] == cluster_labels[j]:
                    weight *= 0.8  # favoriser les mouvements au sein du même cluster
                G.add_edge(i, j, weight=weight)

    # état initial: (bras_gauche, bras_droit, jambe_gauche, jambe_droite, dernier_membre_bougé)
    start_state = (hand_start, hand_start, foot_start, foot_start, None)
    
    # vérifier si une configuration est physiquement possible
    def is_valid_state(state):
        la, ra, ll, rl, _ = state
        positions = [la, ra, ll, rl]
        
        # vérifier la séparation des membres
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if dist_matrix[positions[i], positions[j]] > max_limb_separation:
                    return False
        
        # vérifier le centre de gravité (simplifié)
        center = np.mean([box_centers[p] for p in positions[:4]], axis=0)
        convex_hull = [box_centers[p] for p in [ll, rl]]
        if not any(np.all(center <= point) for point in convex_hull):
            # vérification de stabilité très simplifiée
            return False
        
        return True

    # vérifier si nous avons atteint l'objectif (les deux mains à l'arrivée)
    def is_goal(state):
        la, ra, _, _, _ = state
        return la == finish and ra == finish

    # recherche du meilleur chemin avec a*
    def a_star_search():
        # file de priorité pour l'exploration
        queue = [(0, 0, start_state, [])]  # (priorité, nb_étapes, état, chemin)
        visited = set()
        
        while queue:
            _, steps, state, path = min(queue)
            queue.remove((_, steps, state, path))
            
            if is_goal(state):
                return path + [state]
            
            state_key = state[:4]  # exclure le dernier membre déplacé de la clé
            if state_key in visited:
                continue
            
            visited.add(state_key)
            
            # essayer de déplacer chaque membre vers chaque prise accessible
            la, ra, ll, rl, last_moved = state
            for limb_idx, current_pos in enumerate([la, ra, ll, rl]):
                if limb_idx == last_moved:  # ne pas déplacer le même membre deux fois de suite
                    continue
                    
                for next_hold in G.neighbors(current_pos):
                    # créer un nouvel état en déplaçant un membre
                    new_state = list(state[:4])
                    new_state[limb_idx] = next_hold
                    new_state.append(limb_idx)
                    new_state = tuple(new_state)
                    
                    if new_state[:4] not in visited and is_valid_state(new_state):
                        # calculer la priorité (favoriser moins de mouvements et distances plus courtes)
                        h = min(dist_matrix[new_state[0], finish], dist_matrix[new_state[1], finish])
                        priority = steps + 1 + h/100
                        queue.append((priority, steps + 1, new_state, path + [state]))
        
        return None  # aucun chemin trouvé

    return a_star_search()

def draw_path(img, boxes, box_centers, best_path):
    """dessine le chemin pour chaque membre avec des couleurs différentes"""
    path_img = img.copy()
    
    # couleurs pour chaque membre
    limb_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    limb_names = ['Bras Gauche', 'Bras Droit', 'Jambe Gauche', 'Jambe Droite']
    
    for i in range(len(best_path)-1):
        state = best_path[i]
        next_state = best_path[i+1]
        
        # déterminer quel membre a bougé
        for limb in range(4):
            if state[limb] != next_state[limb]:
                # dessiner une ligne montrant le mouvement du membre
                start_pt = tuple(map(int, box_centers[state[limb]]))
                end_pt = tuple(map(int, box_centers[next_state[limb]]))
                cv2.line(path_img, start_pt, end_pt, limb_colors[limb], 2)
                
                # ajouter le numéro de l'étape
                # décaler légèrement en fonction du type de membre
                text_offset = [(8, 8), (8, -8), (-8, 8), (-8, -8)][limb]
                text_position = (end_pt[0] + text_offset[0], end_pt[1] + text_offset[1])
                # ajouter un rectangle de fond avec la même couleur que le membre
                text_size = cv2.getTextSize(str(i+1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(path_img, 
                             (text_position[0]-2, text_position[1]-text_size[1]-2),
                             (text_position[0]+text_size[0]+2, text_position[1]+2),
                             limb_colors[limb], -1)
                # dessiner le texte légèrement plus grand pour une meilleure visibilité
                cv2.putText(path_img, str(i+1), text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
    
    return path_img, limb_colors, limb_names

def main(image_path):
    # valeurs par défaut
    eps = 118
    max_reach = 120
    max_limb_separation = 200
    
    # charger l'image et les boîtes
    img, boxes, height, width = load_image_and_boxes(image_path)
    
    # afficher les boîtes numérotées pour la sélection
    numbered_img = display_numbered_boxes(img, boxes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.3)
    ax_img = ax.imshow(numbered_img)
    ax.set_title('Sélectionnez les indices de départ et d\'arrivée')
    ax.axis('off')
    
    # créer des zones de texte pour les entrées
    ax_foot = plt.axes([0.25, 0.15, 0.1, 0.05])
    foot_box = TextBox(ax_foot, 'Pied départ:', initial='0')
    
    ax_hand = plt.axes([0.25, 0.08, 0.1, 0.05])
    hand_box = TextBox(ax_hand, 'Main départ:', initial='0')
    
    ax_finish = plt.axes([0.25, 0.01, 0.1, 0.05])
    finish_box = TextBox(ax_finish, 'Arrivée:', initial='0')
    
    # paramètres ajustables
    ax_eps = plt.axes([0.5, 0.15, 0.3, 0.03])
    eps_slider = Slider(ax_eps, 'EPS', 50, 300, valinit=eps)
    
    ax_reach = plt.axes([0.5, 0.10, 0.3, 0.03])
    reach_slider = Slider(ax_reach, 'MAX_REACH', 50, 200, valinit=max_reach)
    
    ax_sep = plt.axes([0.5, 0.05, 0.3, 0.03])
    sep_slider = Slider(ax_sep, 'MAX_LIMB_SEP', 100, 300, valinit=max_limb_separation)
    
    # bouton d'application
    ax_apply = plt.axes([0.5, 0.01, 0.1, 0.03])
    apply_button = Button(ax_apply, 'Appliquer')
    
    # bouton de sortie
    ax_exit = plt.axes([0.7, 0.01, 0.1, 0.03])
    exit_button = Button(ax_exit, 'Quitter')
    
    path_fig = None
    
    def update(_):
        nonlocal path_fig
        
        try:
            foot_start = int(foot_box.text)
            hand_start = int(hand_box.text)
            finish = int(finish_box.text)
            
            # mettre à jour les paramètres
            eps_val = eps_slider.val
            reach_val = reach_slider.val
            sep_val = sep_slider.val
            
            # clustering
            box_centers, cluster_labels, cluster_colors = cluster_boxes(boxes, eps=eps_val)
            
            # dessiner les boîtes avec clustering
            clustered_img = draw_clustered_boxes(img, boxes, cluster_labels, cluster_colors, 
                                              foot_start, hand_start, finish)
            
            # trouver le chemin
            best_path = find_path(boxes, box_centers, cluster_labels, foot_start, hand_start, 
                               finish, max_reach=reach_val, max_limb_separation=sep_val)
            
            if best_path:
                # dessiner le chemin
                path_img, limb_colors, limb_names = draw_path(clustered_img, boxes, box_centers, best_path)
                
                # afficher l'image avec le chemin
                if path_fig is None or not plt.fignum_exists(path_fig.number):
                    path_fig = plt.figure(figsize=(10, 10))
                else:
                    path_fig.clear()
                
                ax_path = path_fig.add_subplot(111)
                ax_path.imshow(path_img)
                
                # créer la légende
                legend_patches = [mpatches.Patch(color=[c/255 for c in limb_colors[i]], 
                                               label=limb_names[i]) 
                                for i in range(4)]
                
                ax_path.legend(handles=legend_patches, loc='upper right')
                ax_path.set_title('Chemin optimal')
                ax_path.axis('off')
                
                path_fig.canvas.draw_idle()
                path_fig.show()
            else:
                print('aucun chemin valide trouvé')
        except ValueError:
            print('veuillez entrer des indices valides')
    
    def exit_program(_):
        plt.close('all')
    
    apply_button.on_clicked(update)
    exit_button.on_clicked(exit_program)
    
    plt.show()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print('usage: python path_finder.py <chemin_image>')
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'erreur: l\'image {image_path} n\'existe pas')
        sys.exit(1)
    
    main(image_path)
