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

def get_clusters(scene) :
    width = len(scene)
    height = len(scene[0])

    # select object of objects in the scene (not (255,255) pixels)
    objects = extract_non_white_pixels(scene)

    # k-means clustring in keypoints
    num_clusters = 3
    clusters = perform_clustering(objects, num_clusters)
        
    return clusters



from skimage.segmentation import slic

def get_superpixels(scene, num_superpixels):
    # Superpixel을 추출합니다.
    segments = slic(scene, n_segments=num_superpixels, compactness=10)

    # Superpixel의 중심점을 찾습니다.
    superpixels = []
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        indices = np.where(mask)
        x_center = int(np.mean(indices[0]))
        y_center = int(np.mean(indices[1]))
        superpixels.append((x_center, y_center))

    return superpixels

import cv2
import numpy as np

def select_superpixels(clusters):
    candidate_list = []
    for cluster in clusters:
        # Extract superpixels from the cluster (you can replace this part with your superpixel algorithm)
        superpixels = extract_superpixels(cluster)  # Replace extract_superpixels with your own implementation
        import pdb; pdb.set_trace()
        # Calculate the center of each superpixel
        centers = []
        for superpixel in superpixels:
            center_x = sum([pixel[0] for pixel in superpixel]) / len(superpixel)
            center_y = sum([pixel[1] for pixel in superpixel]) / len(superpixel)
            centers.append((center_x, center_y))
        
        candidate_list.append(centers)
    
    return candidate_list

def draw_representatives(image, candidate_list):
    for i, candidates in enumerate(candidate_list):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Random color for each cluster
        for candidate in candidates:
            x, y = candidate
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
    
    cv2.imshow("Representatives", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2

def get_center(objects):
    center_x = sum([obj[0] for obj in objects]) / len(objects)
    center_y = sum([obj[1] for obj in objects]) / len(objects)
    return center_x, center_y


def convert_clusters(clusters):
    converted_clusters = []
    for cluster in clusters:
        x, y, rgba = cluster
        converted_clusters.append([x, y, *rgba])
    return converted_clusters

def extract_superpixels(cluster, num_superpixels=100):
    # Create a grayscale image with the same shape as the cluster
    # height, width, _ = cluster.shape
    cluster = convert_clusters(cluster)
    import pdb;pdb.set_trace()
    grayscale_image = cv2.cvtColor(np.array(cluster)[:,2:5], cv2.COLOR_BGR2GRAY)
    import pdb;pdb.set_trace()
    
    # Apply SLIC algorithm to extract superpixels
    slic = cv2.ximgproc.createSuperpixelLSC(grayscale_image, region_size=10)
    slic.iterate()
    
    # Get the labels of the superpixels
    labels = slic.getLabels()
    # Find the center coordinates of each superpixel
    centers = []
    for label in range(slic.getNumberOfSuperpixels()):
        mask = labels == label
        indices = np.where(mask)
        center_x = np.mean(indices[1])
        center_y = np.mean(indices[0])
        centers.append((center_x, center_y))
    
    # Sort the centers based on distance from the image center
    image_center_x, image_center_y = get_center(cluster)
    centers = sorted(centers, key=lambda c: ((c[0] - image_center_x) ** 2 + (c[1] - image_center_y) ** 2))
    
    # Select the specified number of superpixels
    selected_centers = centers[:num_superpixels]
    
    # Create superpixel masks based on the selected centers
    superpixels = []
    for center in selected_centers:
        center_x, center_y = center
        mask = np.logical_and(np.abs(indices[1] - center_x) < 10, np.abs(indices[0] - center_y) < 10)
        superpixels.append(indices[:2][mask])
    
    return superpixels


# Example usage
scene, center_points = get_scene()
clusters = get_clusters(scene)
candidate_list = select_superpixels(clusters)
draw_representatives(scene, candidate_list)
