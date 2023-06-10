import os
import random
import numpy as np
from PIL import Image

def get_scene():
    object_paths = [
        '1-1.jpg',
        '1-2.jpg',
        '1-3.jpg',
        '1-4.jpg',
        '1-5.jpg',
        '2-1.jpg',
        '2-2.jpg',
        '2-3.jpg',
        '2-4.jpg',
        '2-5.jpg',
        '3-1.jpg',
        '3-2.jpg',
        '3-3.jpg',
        '3-4.jpg',
        '3-5.jpg'
    ]

    for i in range(len(object_paths)):
        object_paths[i] = "./objects/" + object_paths[i]

    workspace_width = 480
    workspace_height = 480

    workspace = Image.new('RGBA', (workspace_width, workspace_height), (255, 255, 255, 255))
    selected_objects = []
    selected_objects.append(random.sample([object_paths[0], object_paths[1], object_paths[2], object_paths[3], object_paths[4]], 1)[0])
    selected_objects.append(random.sample([object_paths[5], object_paths[6], object_paths[7], object_paths[8], object_paths[9]], 1)[0])
    selected_objects.append(random.sample([object_paths[10], object_paths[11], object_paths[12], object_paths[13], object_paths[14]], 1)[0])

    used_rectangles = []
    center_points = []

    for object_path in selected_objects:
        object_image = Image.open(object_path).convert('RGBA')
        angle = random.randint(0, 360)
        rotated_object = object_image.rotate(angle, expand=True, fillcolor='white')
        rotated_object = rotated_object.resize((144, 144))

        x = random.randint(0, workspace_width - rotated_object.width)
        y = random.randint(0, workspace_height - rotated_object.height)
        width = rotated_object.width
        height = rotated_object.height

        overlap = True
        while overlap:
            overlap = False
            for rect in used_rectangles:
                if x < rect[0] + rect[2] and x + width > rect[0] and y < rect[1] + rect[3] and y + height > rect[1]:
                    x = random.randint(0, workspace_width - rotated_object.width)
                    y = random.randint(0, workspace_height - rotated_object.height)
                    overlap = True
                    break
        
        used_rectangles.append((x, y, width, height))
        center_points.append([x + rotated_object.width / 2, y + rotated_object.height / 2])
        workspace.paste(rotated_object, (x, y), rotated_object)

    # output_path = 'workspace.png'
    # workspace.save(output_path)
    workspace = np.array(workspace)
    center_points = np.array(center_points)
    
    return workspace, center_points

get_scene()