import cv2
from make_dataset import get_scene
from sklearn.cluster import KMeans


def split_with_kmeans(k, data):
    k = 3
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(cv2.KeyPoint_convert(data))

    labels = kmeans.labels_

    clusters = [[] for _ in range(k)] 

    for i, label in enumerate(labels):
        clusters[label].append(data[i])
    
    return clusters


def draw_keypoints(scene):   
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(scene, None)
    clusters = split_with_kmeans(3, keypoints)
    
    scene_with_keypoints = cv2.drawKeypoints(scene, clusters[0], None, color=(0, 255, 0), flags=0)
    scene_with_keypoints = cv2.drawKeypoints(scene_with_keypoints, clusters[1], None, color=(255, 0, 0), flags=0)
    scene_with_keypoints = cv2.drawKeypoints(scene_with_keypoints, clusters[2], None, color=(0, 0, 255), flags=0)
    
    cv2.imshow('scene_with_keypoints', scene_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
scene, center_points = get_scene()
draw_keypoints(scene)