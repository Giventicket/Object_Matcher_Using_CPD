import cv2
from make_dataset import get_scene
from pycpd import RigidRegistration
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from convex_hull import get_convex_hull
from scipy.optimize import linear_sum_assignment

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

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

for _ in range(100):    
    num_objects = 3

    obs, center_points_gtobs = get_scene()
    goal, center_points_gtgoal = get_scene()

    obs_key_clusters = get_convex_hull(obs)
    goal_key_clusters = get_convex_hull(goal)

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
            callback = partial(visualize, ax=fig.axes[0])

            reg = RigidRegistration(X=target, Y=source)
            # TY, (s_reg, R_reg, t_reg) = reg.register(callback)
            # plt.show()
            
            TY, (s_reg, R_reg, t_reg) = reg.register(None)
            
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