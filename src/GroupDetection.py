import numpy as np
from sklearn.cluster import DBSCAN

def compute_foot_points(boxes):
    foot_points = []
    for box in boxes:
        x1, y1, x2, y2 = box
        foot_X = (x1 + x2) / 2  # center x coordinate
        foot_Y = y2  # bottom y coordinate
        foot_points.append([foot_X, foot_Y])
        
    return np.array(foot_points)

def zone_grouping(foot_points, y_thresh=40, eps_scale=0.5, min_samples=2):
    # Check if foot_points is empty
    if len(foot_points) == 0:
        return np.array([])
        
    # Sıralı y-koordinatlara göre zone'lara ayır
    sorted_indices = np.argsort(foot_points[:, 1])
    foot_points_sorted = foot_points[sorted_indices]
    
    zones = []
    current_zone = [foot_points_sorted[0]]
    
    for i in range(1, len(foot_points_sorted)):
        if abs(foot_points_sorted[i][1] - foot_points_sorted[i-1][1]) > y_thresh:
            zones.append(np.array(current_zone))
            current_zone = []
        current_zone.append(foot_points_sorted[i])
    zones.append(np.array(current_zone))

    group_labels = np.full(len(foot_points), -1)  # default olarak -1

    label_counter = 0
    for zone in zones:
        if len(zone) >= min_samples:
            eps = np.mean([np.linalg.norm(p1 - p2) for p1 in zone for p2 in zone]) * eps_scale
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(zone)
            for i, label in enumerate(clustering.labels_):
                if label != -1:
                    global_idx = np.where((foot_points == zone[i]).all(axis=1))[0][0]
                    group_labels[global_idx] = label + label_counter
            label_counter += max(clustering.labels_, default=0) + 1
    return group_labels
