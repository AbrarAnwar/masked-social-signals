import cv2
import numpy as np
import math
import torch


def create_image(width, height):
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image[:] = (255, 255, 255)
    return blank_image

def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y


def visualize_headgaze(image, est_gaze,color=(255,0,0)):
    output_image = np.copy(image)
    center_x = output_image.shape[1] / 2
    center_y = output_image.shape[0] / 2

    endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 100)

    cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), color, 2)
    bordered_image = cv2.copyMakeBorder(output_image, top=5, bottom=5, left=5, right=5, 
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered_image


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



def write_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()


def construct_batch_video(batch, task, color=(255,0,0)):
    # batch shape (16, 3, 180, 2)
    batch_frames = []
    for i in range(batch.shape[0]):
        each_bacth = batch[i]
        frames = []

        for j in range(each_bacth.shape[1]):
            people_image = []

            for k in range(each_bacth.shape[0]):
                person = each_bacth[k][j]
                
                if task == 'pose':
                    blank = create_image(600, 500)
                    image = visualize_pose(blank, person)
                else:
                    blank = create_image(250, 250)
                    image = visualize_headgaze(blank, person, color)

                people_image.append(image)

            frames.append(np.concatenate(people_image, axis=1))
        batch_frames.append(frames)

    return np.array(batch_frames)



#if __name__ == '__main__':
    #frames = []
    #for est in gaze[:180]:
    #    blank = create_image(250,250)
    #    image = visualize_headgaze(blank, est)
    #    frames.append(image)
    #write_video(frames, 'test.mp4')

