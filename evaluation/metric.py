import torch
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance


class PCK:
    def __init__(self, alphas=[0.1, 0.08, 0.05]):
        """
        Initialize the PCK metric tracker.

        Parameters:
        - thresholds: List of threshold values to compute PCK over multiple thresholds.
        """
        self.alphas = alphas
        self.all_pred_keypoints = []
        self.all_gt_keypoints = []
        self.total_keypoints = 0  # to keep track of total keypoints across batches
        self.updated = False
    

    def get_thresh(self, alpha, gt):
        """
        Computes the threshold for PCK based on an alpha ratio between the min and max values of the ground truth keypoints.

        Parameters:
        - alpha: The ratio used to compute the threshold.
        - gt: (batch_size, 3, temporal, 26) ground truth keypoints.

        Returns:
        - thresh: The computed threshold for PCK.
        """
        
        # Reshape ground truth keypoints to (batch_size, 3, temporal, 13, 2) to separate x and y coordinates
        gt_reshaped = gt.reshape(gt.shape[0], 3, gt.shape[2], 13, 2)

        # Compute the width (x max - x min) and height (y max - y min)
        w = np.max(gt_reshaped, axis=-2)[..., 0] - np.min(gt_reshaped, axis=-2)[..., 0]  # Width for each keypoint
        h = np.max(gt_reshaped, axis=-2)[..., 1] - np.min(gt_reshaped, axis=-2)[..., 1]  # Height for each keypoint

        thresh = alpha * np.max(np.stack([w, h], axis=-1), axis=-1, keepdims=True)  # Shape: (batch_size, 3, temporal, 13)

        return thresh
    

    def update(self, predicted_keypoints, ground_truth_keypoints):
        """
        Update the metric with a new batch of data.

        Parameters:
        - predicted_keypoints: (batch_size, 3, temporal, 26) array of predicted normalized keypoints.
        - ground_truth_keypoints: (batch_size, 3, temporal, 26) array of ground truth normalized keypoints.
        """
        # Append the batch predictions and ground truths to the lists
        self.updated = True
        self.all_pred_keypoints.append(predicted_keypoints.cpu().numpy())
        self.all_gt_keypoints.append(ground_truth_keypoints.cpu().numpy())

        self.total_keypoints += predicted_keypoints.shape[0] * 3 * predicted_keypoints.shape[2] * 13
    

    def compute(self):
        """
        Compute the final micro-average PCK over all batches and all thresholds.

        Returns:
        - average_pck: Average PCK across all thresholds.
        - pck_per_threshold: PCK for each threshold.
        """
        if not self.updated:
            return 0.0
        
        # Concatenate all collected batches along the batch dimension
        all_pred_keypoints = np.concatenate(self.all_pred_keypoints, axis=0)
        all_gt_keypoints = np.concatenate(self.all_gt_keypoints, axis=0)

        # Reshape to (total_samples, 3, temporal, 13, 2) for easier distance calculation
        pred_reshaped = all_pred_keypoints.reshape(all_pred_keypoints.shape[0], 3, all_pred_keypoints.shape[2], 13, 2)
        gt_reshaped = all_gt_keypoints.reshape(all_gt_keypoints.shape[0], 3, all_gt_keypoints.shape[2], 13, 2)

        # Compute the Euclidean distance between predicted and ground truth keypoints
        distances = np.linalg.norm(pred_reshaped - gt_reshaped, axis=-1)  # Shape: (total_samples, 3, temporal, 13)

        # Initialize list to store PCK results for each threshold
        pck_per_threshold = []

        # Loop through each threshold and calculate the PCK
        for alpha in self.alphas:
            threshold = self.get_thresh(alpha, all_gt_keypoints)
            correct_keypoints = distances < threshold  # boolean mask of correct keypoints within the threshold
            total_correct_keypoints = np.sum(correct_keypoints)  # total number of correct keypoints across all samples
            pck_batch = total_correct_keypoints / self.total_keypoints  # micro-average PCK

            pck_per_threshold.append(pck_batch)

        # Calculate the average PCK across all thresholds
        average_pck = np.mean(pck_per_threshold)

        return average_pck #, pck_per_threshold
    

class FID:
    def __init__(self):
        """
        Initializes the FID metric calculator.
        
        Parameters:
        - num_features: The number of features for each sample (26 for keypoints, 2 for gaze/headpose).
        """
        self.all_pred = []
        self.all_gt = []
        self.updated = False

    def update(self, predicted, ground_truth):
        """
        Update the metric with new batches of predicted and ground truth features.
        
        Parameters:
        - predicted: (batch_size, 3, temporal, num_features) array of predicted features.
        - ground_truth: (batch_size, 3, temporal, num_features) array of ground truth features.
        """
        self.updated = True
        self.all_pred.append(predicted.cpu().numpy())
        self.all_gt.append(ground_truth.cpu().numpy())

    def compute(self):
        """
        Computes the FID score between the predicted and ground truth features.

        Returns:
        - fid: The computed FID score.
        """
        if not self.updated:
            return 0.0
        
        # Concatenate all batches along the batch dimension
        pred = np.concatenate(self.all_pred, axis=0)
        gt = np.concatenate(self.all_gt, axis=0)

        # (total_samples*3, temporal*feature_dim)
        # Flatten the features to 2D (samples, features) for FID calculation
        pred_flat = pred.reshape(pred.shape[0]*3*pred.shape[2], -1)
        gt_flat = gt.reshape(gt.shape[0]*3*pred.shape[2], -1)

        # Compute mean and covariance of predicted features
        mu_pred = np.mean(pred_flat, axis=0)
        sigma_pred = np.cov(pred_flat, rowvar=False)

        # Compute mean and covariance of ground truth features
        mu_gt = np.mean(gt_flat, axis=0)
        sigma_gt = np.cov(gt_flat, rowvar=False)

        # Compute the squared difference of the means
        diff = mu_pred - mu_gt
        diff_squared = np.sum(diff ** 2)

        # Compute the product of the covariance matrices and its square root
        cov_mean = sqrtm(sigma_pred.dot(sigma_gt))

        # Numerical stability: if there are any imaginary components due to sqrtm, set them to 0
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        # Compute the FID score
        fid = diff_squared + np.trace(sigma_pred + sigma_gt - 2 * cov_mean)
        return fid
    


class W1:
    def __init__(self):
        """
        Initializes the W1 metric calculator for velocity and acceleration.
        
        """
        self.all_pred = []
        self.all_gt = []
        self.updated = False

    def update(self, predicted, ground_truth):
        """
        Update the metric with new batches of predicted and ground truth features.
        
        Parameters:
        - predicted: (batch_size, 3, temporal, num_features) array of predicted features.
        - ground_truth: (batch_size, 3, temporal, num_features) array of ground truth features.
        """
        self.updated = True
        self.all_pred.append(predicted.cpu().numpy())
        self.all_gt.append(ground_truth.cpu().numpy())

    def compute(self):
        """
        Computes the W1 metric for velocity and acceleration between the predicted and ground truth features.

        Returns:
        - w1_vel: The W1 distance for velocity.
        - w1_acc: The W1 distance for acceleration.
        """
        # Concatenate all batches along the batch dimension
        if not self.updated:
            return (0.0, 0.0)
        

        feature_dim = self.all_pred[0].shape[-1]
        
        pred = np.concatenate(self.all_pred, axis=0)  # (batch_size, 3, temporal, num_features)
        gt = np.concatenate(self.all_gt, axis=0)
        bz = pred.shape[0]
        # Combine batch size and number of people into a single dimension

        # Reshape temporal and feature dimensions
        pred_flat = pred.reshape(bz*3, pred.shape[2], feature_dim)  # (batch_size * 3, temporal, num_features)
        gt_flat = gt.reshape(bz*3, gt.shape[2], feature_dim)

        # Compute velocity as first derivative (difference along the temporal axis)
        vel_pred = np.diff(pred_flat, axis=1)  # Shape: (batch_size * 3, temporal - 1, num_features)
        vel_gt = np.diff(gt_flat, axis=1)

        # Compute acceleration as second derivative (difference of velocities)
        acc_pred = np.diff(vel_pred, axis=1)  # Shape: (batch_size * 3, temporal - 2, num_features)
        acc_gt = np.diff(vel_gt, axis=1)

        # Initialize W1 scores
        w1_vel_total = 0
        w1_acc_total = 0

        # Compute W1 for each feature dimension
        for i in range(feature_dim):
            # W1 distance for velocity
            for j in range(vel_pred.shape[0]):  # Loop through each person in the batch
                w1_vel_total += wasserstein_distance(vel_pred[j, :, i], vel_gt[j, :, i])

            # W1 distance for acceleration
            for j in range(acc_pred.shape[0]):
                w1_acc_total += wasserstein_distance(acc_pred[j, :, i], acc_gt[j, :, i])

        # Average W1 scores by the number of people and features
        w1_vel = w1_vel_total / (bz * 3 * feature_dim)
        w1_acc = w1_acc_total / (bz * 3 * feature_dim)

        return (w1_vel, w1_acc)


class L2:
    def __init__(self):
        """
        Initializes the L2 loss accumulator for micro-average loss calculation.
        """
        self.total_loss = 0.0  # To accumulate the sum of losses over batches
        self.total_samples = 0  # To track the total number of samples across batches
        self.updated = False

    def update(self, predicted, ground_truth):
        """
        Update the L2 loss with new batches of predicted and ground truth features.

        Parameters:
        - predicted: (batch_size, 3, temporal, num_features) tensor of predicted features.
        - ground_truth: (batch_size, 3, temporal, num_features) tensor of ground truth features.
        """
        self.updated = True
        # Compute the mean squared error (MSE) for the current batch
        loss = torch.nn.functional.mse_loss(predicted, ground_truth, reduction='sum')  # Sum to accumulate across all elements

        # Update total loss
        self.total_loss += loss.item()

        # get the total number of samples in the batch
        self.total_samples += predicted.numel()

    def compute(self):
        """
        Compute the final micro-average L2 loss across all batches.
        
        Returns:
        - The micro-average L2 loss.
        """
        if not self.updated:
            return 0.0
        
        # Compute the micro-average loss by dividing the total loss by the total number of samples
        micro_average_loss = self.total_loss / self.total_samples
        return micro_average_loss


