import cv2
from make_dataset import get_scene
from pycpd import RigidRegistration
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_keypoints(scene):   
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(scene, None)
    scene_with_keypoints = cv2.drawKeypoints(scene, keypoints, None, color=(0, 255, 0), flags=0)
    
    return cv2.KeyPoint_convert(keypoints)

def split_with_kmeans(k, data):
    k = 3
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)

    labels = kmeans.labels_

    clusters = [[] for _ in range(k)] 

    for i, label in enumerate(labels):
        clusters[label].append(data[i])
    
    return clusters

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

obs, center_points_obs = get_scene()
goal, center_points_goal = get_scene()

obs_keys = get_keypoints(obs)
goal_keys = get_keypoints(goal)

obs_keys = split_with_kmeans(3, obs_keys)
goal_keys = split_with_kmeans( 3, goal_keys)

for i, (source, target) in enumerate(zip(obs_keys, goal_keys)):
    obs_keys[i] = np.array(source)
    goal_keys[i] = np.array(target)

permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

best_score = 100000000
match = None

for permutation in permutations:
    score = 0
    for i, source in enumerate(obs_keys):
        obs_keys[i] = np.array(source)
        target = goal_keys[permutation[i]]

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
    
    print(score)
    
    if score < best_score:
        best_score = score
        match = permutation
        
print(best_score, match)
     