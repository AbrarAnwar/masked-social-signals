import cv2
import numpy as np
import math
import torch

from utils.visualize_pose import *

path = '/home/tangyimi/social_signal/dining_dataset/full_gazes/01_1.npz'

# Load the data
data = np.load(path)
gaze = data['headpose'] # shape (180, 2)


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



def evaluate(model, batch, task):
    # batch (16,3,1080,26)
    if task == 'pose':
        batch = index_pose(batch)

    bz = batch.size(0)
    batch = batch.reshape(bz, 3, 6, 180, batch.size(-1))

    # do segement
    x = batch[:, :, :5, :, :]
    y = batch[:, :, 5, :, :] # shape (bz, 3, 180, 2)
    
    # reshape
    x = x.permute(0, 2, 1, 3, 4).reshape(bz, 5, -1)
    x_flatten = x.view(x.size(0), -1)
    
    model.eval()
    with torch.no_grad():
        y_hat = model.forward(x_flatten) # shape (bz, 3*180*feature_dim)

    y_hat = y_hat.reshape(bz, 3, 180, -1)

    videos_prediction = construct_batch_video(y_hat, task=task)
    videos_inference = construct_batch_video(y, task=task, color=(0,0,255))

    result_videos = np.concatenate([videos_prediction, videos_inference], axis=2) # shape (bz, 180, 520, 780, 3)

    return result_videos



if __name__ == '__main__':
    frames = []
    for est in gaze[:180]:
        blank = create_image(250,250)
        image = visualize_headgaze(blank, est)
        frames.append(image)
    write_video(frames, 'test.mp4')
