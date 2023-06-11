import cv2
from make_dataset import get_scene
from pycpd import RigidRegistration
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# pred
def get_keypoints(scene):   
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(scene, None)
    scene_with_keypoints = cv2.drawKeypoints(scene, keypoints, None, color=(0, 255, 0), flags=0)
    
    return cv2.KeyPoint_convert(keypoints)

def split_with_kmeans(k, data):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    clusters = [[] for _ in range(k)] 
    for i, label in enumerate(labels):
        clusters[label].append(data[i])
    return clusters


def visualize(iteration, error, X, Y, ax, goal):
    plt.cla()
    ax.imshow(goal)
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target', s=5)
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source', s=5)
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


# for evaluation metric
def get_center_points_from_keypoints(clusters):
    center_points = np.zeros((len(clusters), 2))
    for i, cluster in  enumerate(clusters):
        center_points[i, :] = cluster.mean(0)
    return center_points

def find_matching_indices(pred, gt):
    # Calculate pairwise distances between all observations
    pairwise_distances = np.linalg.norm(gt[:, np.newaxis] - pred, axis=2)
    # Find the indices that minimize the sum of pairwise distances
    row_indices, col_indices = linear_sum_assignment(pairwise_distances)
    return col_indices


for _ in range(0, 100):
    num_objects = 3

    obs, center_points_gtobs = get_scene()
    goal, center_points_gtgoal = get_scene()

    obs_keys = get_keypoints(obs)
    goal_keys = get_keypoints(goal)

    obs_key_clusters = split_with_kmeans(num_objects, obs_keys)
    goal_key_clusters = split_with_kmeans(num_objects, goal_keys)


    for i, (source, target) in enumerate(zip(obs_key_clusters, goal_key_clusters)):
        obs_key_clusters[i] = np.array(source) 
        goal_key_clusters[i] = np.array(target)

    permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

    best_score = 100000000
    match = None

    for permutation in permutations:
        score = 0
        for i, source in enumerate(obs_key_clusters):
            target = goal_key_clusters[permutation[i]]

            fig = plt.figure()
            fig.add_axes([0, 0, 1, 1])
            callback = partial(visualize, ax=fig.axes[0], goal=goal)

            reg = RigidRegistration(X=target, Y=source)
            TY, (s_reg, R_reg, t_reg) = reg.register(callback)
            plt.show()
            
            # TY, (s_reg, R_reg, t_reg) = reg.register(None)
            
            # TY is the transformed source points
            #  s_reg the scale of the registration
            #  R_reg the rotation matrix of the registration
            #  t_reg the translation of the registration    

            score = score + reg.q
        
        # print(score)
        
        if score < best_score:
            best_score = score
            match = permutation
            
    # print(best_score, match)

    center_points_obs = get_center_points_from_keypoints(obs_key_clusters)
    center_points_goal = get_center_points_from_keypoints(goal_key_clusters)

    # print(center_points_obs)
    # print(center_points_gtobs)
    # print(find_matching_indices(center_points_obs, center_points_gtobs))
    obs_mapping = find_matching_indices(center_points_obs, center_points_gtobs)

    # print(center_points_goal)
    # print(center_points_gtgoal)
    # print(find_matching_indices(center_points_goal, center_points_gtgoal))
    goal_mapping = find_matching_indices(center_points_goal, center_points_gtgoal)

    answer = []
    for i in range(num_objects):
        answer.append([obs_mapping[i], goal_mapping[i]])
    answer.sort()

    pred = []
    for i in range(num_objects):
        pred.append([i, match[i]])
    pred.sort()


    # print(answer)
    # print(pred)
    if answer == pred:
        print("correct")
    else:
        print("wrong")