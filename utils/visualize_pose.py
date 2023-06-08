import numpy as np
import torch
import cv2

pose = np.array([274.612,168.87,0.808275,307.258,188.427,0.598458,214.558,177.99,0.573352,183.234,271.992,0.560101,167.555,349.007,0.6798,403.79,202.775,0.448373,369.855,363.358,0.486654,227.595,388.147,0.182503,264.144,412.92,0.122994,209.333,390.734,0.160456,0,0,0,0,0,0,321.581,418.165,0.137722,0,0,0,0,0,0,262.819,129.713,0.807337,308.566,150.533,0.852742,244.577,106.192,0.138878,365.929,142.778,0.823036,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

POSE_INDEX = [0,1,2,3,4,5,6,7,8,15,16,17,18]

def get_xy(pose, index):
    return (int(pose[index*2]), int(pose[index*2+1]))


def visualize_pose(image, pose):  
    thickness = 4 

    cv2.line(image, get_xy(pose, 0), get_xy(pose, 1), (51,0,153), thickness)
    cv2.line(image, get_xy(pose, 0), get_xy(pose, 9), (102,0,153), thickness)
    cv2.line(image, get_xy(pose, 0), get_xy(pose, 10), (153,0,102), thickness)
    cv2.line(image, get_xy(pose, 1), get_xy(pose, 2), (1,51,153), thickness)
    cv2.line(image, get_xy(pose, 1), get_xy(pose, 5), (0,153,102), thickness)
    cv2.line(image, get_xy(pose, 1), get_xy(pose, 8), (1,0,153), thickness)
    cv2.line(image, get_xy(pose, 2), get_xy(pose, 3), (1,102,154), thickness)
    cv2.line(image, get_xy(pose, 3), get_xy(pose, 4), (0,153,153), thickness)
    cv2.line(image, get_xy(pose, 5), get_xy(pose, 6), (0,153,51), thickness)
    cv2.line(image, get_xy(pose, 6), get_xy(pose, 7), (0,153,0), thickness)
    cv2.line(image, get_xy(pose, 9), get_xy(pose, 11), (153,0,153), thickness)
    cv2.line(image, get_xy(pose, 10), get_xy(pose, 12), (153,0,51), thickness)
    bordered_image = cv2.copyMakeBorder(image, top=5, bottom=5, left=5, right=5, 
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered_image


def index_pose(batch):
     # 9, 10, 11, 12
    indices = [0,1,2,3,4,5,6,7,8,15,16,17,18]
    indices_xy = [j for i in indices for j in (i*3, i*3 + 1)] 

    indices_xy = torch.tensor(indices_xy).to(batch.device)

    batch_selected = batch.index_select(3, indices_xy)

    return batch_selected
    
    
