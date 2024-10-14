import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.dataset import get_loaders

SKELETON_PAIRS = [(0,1), (0,9), (0,10), (1,2), (1,5), (1,8), (2,3), (3,4), (5,6), (6,7), (9,11), (10,12)]


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
        each_batch = batch[i]
        frames = []

        for j in range(each_batch.shape[1]):
            people_image = []

            for k in range(each_batch.shape[0]):
                person = each_batch[k][j]
                
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


def visualize(task, normalizer, y, y_hat, file_name):
    output = 0
    fps = 15 if task == 'pose' else 30
    current_y = normalizer.minmax_denormalize(y, task)
    current_y_hat = normalizer.minmax_denormalize(y_hat, task)

    videos_prediction = construct_batch_video(current_y_hat, task=task)
    videos_truth = construct_batch_video(current_y, task=task, color=(0,0,255))

    result_videos = np.concatenate([videos_prediction, videos_truth], axis=2)

    for i in range(current_y.size(0)):
        new_file_name = file_name + '_' + str(output) + '.mp4'
        write_video(result_videos[i], new_file_name, fps)
        output += 1



def plot(pred_batch, gt_batch, save_path):
    # Create a figure with 2 rows (predictions on top, ground truth on bottom) and 3 columns (one for each person)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    
    # List of batches for iteration (top row: predictions, bottom row: ground truth)
    batches = [(pred_batch, 'Prediction'), (gt_batch, 'Ground Truth')]

    for row_idx, (batch, title) in enumerate(batches):

        for b in range(pred_batch.size(0)):
            for person_idx in range(3):
                # Extract all frames for the current person
                person_keypoints = batch[person_idx, :, :]  # Shape (temporal, 26)

                ax[row_idx, person_idx].invert_yaxis()
                ax[row_idx, person_idx].set_xticks([])
                ax[row_idx, person_idx].set_yticks([])
                #ax[row_idx, person_idx].set_title(f'{title} Person {person_idx + 1}')
                
                # Plot skeleton and scatter points for all frames
                for temporal_idx in range(person_keypoints.shape[0]):
                    frame_keypoints = person_keypoints[temporal_idx, :]
                    
                    # Plot skeleton lines
                    for first, second in SKELETON_PAIRS:
                        first_list = [get_xy(frame_keypoints, first)[0], get_xy(frame_keypoints, second)[0]][::-1]
                        second_list = [get_xy(frame_keypoints, first)[1], get_xy(frame_keypoints, second)[1]][::-1]
                        ax[row_idx, person_idx].plot(first_list, second_list, color='black', linewidth=0.1, alpha=0.2)
                    
                    # Plot hands
                    ax[row_idx, person_idx].scatter(get_xy(frame_keypoints, 4)[0], get_xy(frame_keypoints, 4)[1], color='red', s=10, alpha=0.4)  # Left hand
                    ax[row_idx, person_idx].scatter(get_xy(frame_keypoints, 7)[0], get_xy(frame_keypoints, 7)[1], color='blue', s=10, alpha=0.4)  # Right hand
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



# if __name__ == '__main__':
#     _, _, test_loader = get_loaders(batch_path='./dining_dataset/batch_window36_stride18_v4', 
#                                     test_idx=30)
    
#     for batch_idx, batch in enumerate(test_loader):
#         if batch_idx == 10:
#             plot(batch['pose'][7], batch['pose'][7], f'./pose.png')
#             break

