import cv2
from make_dataset import get_scene

def draw_keypoints(scene):   
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(scene, None)
    
    scene_with_keypoints = cv2.drawKeypoints(scene, keypoints, None, color=(0, 255, 0), flags=0)
    
    cv2.imshow('scene_with_keypoints', scene_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

scene, centers = get_scene()
draw_keypoints(scene)
