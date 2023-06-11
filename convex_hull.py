import cv2
import numpy as np
from make_dataset import get_scene
import numpy as np
from sklearn.cluster import KMeans

# select non_white_pixels
def extract_non_white_pixels(workspace):
    non_white_pixels = []
    height, width, _ = workspace.shape

    for x in range(height):
        for y in range(width):
            pixel = workspace[x][y]
            if not np.array_equal(pixel, [255, 255, 255, 255]):
                non_white_pixels.append((x, y, tuple(pixel)))

    return non_white_pixels

# k-means clustering
def perform_clustering(keypoints, num_clusters):
    # 위치 정보와 색 정보를 추출하여 배열로 변환합니다.
    # data = np.array([(x, y, r, g, b, a) for x, y, (r, g, b, a) in keypoints])
    data = np.array([(x, y) for x, y, (r, g, b, a) in keypoints])
    
    # K-means 클러스터링을 수행합니다.
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    # 클러스터 할당 결과를 가져옵니다.
    labels = kmeans.labels_

    # 클러스터링 결과를 저장할 리스트를 초기화합니다.
    clusters = [[] for _ in range(num_clusters)]

    # 각 키포인트를 해당하는 클러스터에 할당합니다.
    for i, label in enumerate(labels):
        clusters[label].append(keypoints[i])

    return clusters

# display clusters results
def display_clusters(scene,clusters, width, height):
    # 이미지를 생성합니다.
    image = np.zeros((height, width, 3), np.uint8)

    # 클러스터에 대해 색상을 할당합니다.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # 각 점을 이미지에 그립니다.
    for cluster_idx, cluster in enumerate(clusters):
        # for x, y in cluster:
        for x, y, _ in cluster:
            color = colors[cluster_idx]
            image[x, y] = color

    # 이미지를 화면에 표시합니다.
    cv2.imshow('Clustered Image', image)
    cv2.imshow('Original Image', scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def display_and_return_object_boundaries(scene, clusters, width, height):
    # 이미지를 생성합니다.
    image = np.copy(scene)
    
    # 객체 볼록 다각형 경계를 그릴 색상을 정의합니다.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    convex_hull_list = [[] for _ in range(3)]
    
    # 각 객체의 볼록 다각형 경계를 그립니다.
    for cluster_idx, cluster in enumerate(clusters):
        # 객체의 위치 정보를 추출합니다.
        points = [(int(y), int(x)) for x, y, _ in cluster]
        
        
        # 볼록 다각형 경계를 계산합니다.
        hull = cv2.convexHull(np.array(points))
        hull_list = hull.tolist()
        squeezed_hull_list = [sublist[0] for sublist in hull_list]
        convex_hull_list[cluster_idx].append(squeezed_hull_list)
        
        # 경계를 그립니다.
        color = colors[cluster_idx]
        cv2.drawContours(image, [hull], 0, color, thickness=2)

    # 이미지를 화면에 표시합니다.
    # cv2.imshow('Object Boundaries', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # squeeze
    convex_hull_list = [sublist[0] for sublist in convex_hull_list]
    
    return convex_hull_list

def get_convex_hull(scene) :
    width = len(scene)
    height = len(scene[0])

    # select object of objects in the scene (not (255,255) pixels)
    objects = extract_non_white_pixels(scene)

    # k-means clustring in keypoints
    num_clusters = 3
    clusters = perform_clustering(objects, num_clusters)

    # display_clusters(scene,clusters,width,height)
    convex_hull_list = display_and_return_object_boundaries(scene, clusters, width, height)
    
    return convex_hull_list

# scene, center_points = get_scene()
# cv2.imshow('original Boundaries', scene)
# get_convex_hull(scene)