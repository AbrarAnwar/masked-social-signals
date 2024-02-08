from tqdm import tqdm
import torch
import numpy as np
import scipy
from scipy import linalg


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()
    
  def reset(self):
    self.val = 0
    self.avg = torch.Tensor([0])[0]
    self.sum = 0
    self.count = 0
    self.val2 = 0
    self.sum_energy = 0
    self.avg_energy = 0
    
  def update(self, val, n=1, val2=None):
    self.count += n
    self.val = val
    self.sum += val * n
    self.avg = self.sum / self.count
    self.val2 = val2
    if val2 is not None:
      self.sum_energy += val2 * n
      self.avg_energy = self.sum_energy / self.count
      
  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)


class PCK():
  '''Computes PCK for different values of alpha and for each joint and returns it as a dictionary'''
  def __init__(self, alphas=[0.1, 0.2, 0.08, 0.05], num_joints=13):
    self.alphas = alphas
    self.num_joints = num_joints
    self.avg_meters = {'pck_{}_{}'.format(al, jnt):AverageMeter('pck_{}_{}'.format(al, jnt)) for al in alphas for jnt in range(num_joints)}
    self.avg_meters.update({'pck_{}'.format(alpha):AverageMeter('pck_{}'.format(alpha)) for alpha in self.alphas})
    self.avg_meters.update({'pck':AverageMeter('pck')})

  '''
  y:  (B, 3, 45, 26) -> (B*3*45, 2, joints)
  gt: (B, 3, 45, 26) -> (B*3*45, 2, joints)
  '''
  def __call__(self, y, gt):
    B = y.shape[0]
    # (b,3,45,26)
    y = y.view(y.size(0) * y.size(1) * y.size(2), 2, -1)
    gt = gt.view(gt.size(0) * gt.size(1) * gt.size(2), 2, -1)

    dist = (((y - gt)**2).sum(dim=1)**0.5)
    pck_avg = 0
    for alpha in self.alphas:
      thresh = self.get_thresh(gt, alpha)
      pck = self.pck(dist, thresh)
      for jnt in range(self.num_joints):
        key = 'pck_{}_{}'.format(alpha, jnt)
        self.avg_meters[key].update(pck.mean(dim=0)[jnt], n=B)

      pck_alpha = pck.mean()
      self.avg_meters['pck_{}'.format(alpha)].update(pck_alpha, n=B)
      pck_avg += pck_alpha

    self.avg_meters['pck'].update(pck_avg/len(self.alphas), n=B)
    
    # for alpha in self.alphas:
    #   self.avg_meters['pck'].update(self.avg_meters['pck_{}'.format(alpha)].avg, n=B)

      
  def pck(self, dist, thresh):
    return (dist < thresh).to(torch.float)
    
  def get_thresh(self, gt, alpha):
    h = gt[:, 0, :].max(dim=-1).values - gt[:, 0, :].min(dim=-1).values
    w = gt[:, 1, :].max(dim=-1).values - gt[:, 1, :].min(dim=-1).values
    thresh = alpha * torch.max(torch.stack([h, w], dim=-1), dim=-1, keepdim=True).values
    return thresh

  def get_averages(self):
    averages = {}
    for alpha in self.alphas:
      for jnt in range(self.num_joints):
        key = 'pck_{}_{}'.format(alpha, jnt)
        averages.update({key:self.avg_meters[key].avg.item()})

      key = 'pck_{}'.format(alpha)
      averages.update({key:self.avg_meters[key].avg.item()})
    key = 'pck'
    averages.update({key:self.avg_meters[key].avg.item()})
    return averages

  def reset(self):
    for key in self.avg_meters:
      self.avg_meters[key].reset()


class FID():
  def __init__(self):
    self.gt_sum_meter = AverageMeter('gt_sum')
    self.gt_square_meter = AverageMeter('gt_square')
    self.y_sum_meter = AverageMeter('y_sum')
    self.y_square_meter = AverageMeter('y_square')

  def __call__(self, y, gt):
    y = y.reshape(-1, y.size(-1)) ## (B, 3, 90, 2) -> (B*3*90, 2) 
    gt = gt.reshape(-1, gt.size(-1)) ## (B, 3, 90, 2) -> (B*3*90, 2) 
    self.gt_sum_meter.update(gt.mean(0, keepdim=True), n=gt.shape[0])
    self.y_sum_meter.update(y.mean(0, keepdim=True), n=y.shape[0])
    self.gt_square_meter.update(gt.T.matmul(gt)/gt.shape[0], n=gt.shape[0])
    self.y_square_meter.update(y.T.matmul(y)/y.shape[0], n=y.shape[0])
    
  def reset(self):
    self.gt_sum_meter.reset()
    self.y_sum_meter.reset()
    self.gt_square_meter.reset()
    self.y_square_meter.reset()

  def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance. 
    Borrowed from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
              inception net (like returned by the function 'get_predictions')
              for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
              representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
              representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
      'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
      'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
      msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
      print(msg)
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
      if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        m = np.max(np.abs(covmean.imag))
        raise ValueError('Imaginary component {}'.format(m))
      covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

  def get_averages(self):
    try:
      N = self.gt_sum_meter.count
      gt_mu = self.gt_sum_meter.avg.squeeze()
      y_mu = self.y_sum_meter.avg.squeeze()

      gt_sum = self.gt_sum_meter.sum
      y_sum = self.y_sum_meter.sum

      gt_square = self.gt_square_meter.sum
      y_square = self.y_square_meter.sum

      gt_cross = gt_sum.T.matmul(gt_sum)
      y_cross = y_sum.T.matmul(y_sum)

      gt_sigma = (gt_square - gt_cross/N)/(N-1)
      y_sigma = (y_square - y_cross/N)/(N-1) ## divide by N-1 for no bias in the estimator

      fid = self.calculate_frechet_distance(gt_mu.detach().cpu(), gt_sigma.detach().cpu(), y_mu.detach().cpu(), y_sigma.detach().cpu())
    except:
      fid = 1000
    return {'FID':fid}


class W1():
  def __init__(self):
    self.gt_vel_meter = AverageMeter('gt_vel')
    self.gt_acc_meter = AverageMeter('gt_acc')
    self.y_vel_meter = AverageMeter('y_vel')
    self.y_acc_meter = AverageMeter('y_acc')
    self.ranges = np.arange(0, 300, 0.1)
    
  def get_vel_acc(self, y):
    diff = lambda x:x[:, :, 1:] - x[:, :, :-1]
    absolute = lambda x:((x**2).sum(3)**0.5).mean(-1).view(-1)
    vel = diff(y)
    acc = diff(vel)
    vel = absolute(vel)  ## average speed accross all joints
    acc = absolute(acc)
    return vel.detach().cpu(), acc.detach().cpu()
    
  def __call__(self, y, gt):
    y = y.reshape(y.size(0), y.size(1), y.size(2), 2, -1)
    gt = gt.reshape(gt.size(0), gt.size(1), gt.size(2), 2, -1) 

    y_vel, y_acc = self.get_vel_acc(y)
    gt_vel, gt_acc = self.get_vel_acc(gt)

    ## make histogram
    y_vel, _ = np.histogram(y_vel, bins=self.ranges)
    y_acc, _ = np.histogram(y_acc, bins=self.ranges)
    gt_vel, _ = np.histogram(gt_vel, bins=self.ranges)
    gt_acc, _ = np.histogram(gt_acc, bins=self.ranges)
    
    self.y_vel_meter.update(y_vel, n=1)
    self.y_acc_meter.update(y_acc, n=1)
    self.gt_vel_meter.update(gt_vel, n=1)
    self.gt_acc_meter.update(gt_acc, n=1)

  def reset(self):
    self.y_vel_meter.reset()
    self.y_acc_meter.reset()
    self.gt_vel_meter.reset()
    self.gt_acc_meter.reset()

  def get_averages(self):
    N = self.ranges[:-1]
    try:
      W1_vel = scipy.stats.wasserstein_distance(N, N,
                                                self.y_vel_meter.sum,
                                                self.gt_vel_meter.sum)
      W1_acc = scipy.stats.wasserstein_distance(N, N,
                                                self.y_acc_meter.sum,
                                                self.gt_acc_meter.sum)
    except:
      W1_vel = 1000
      W1_acc = 1000
      
    return {'W1_vel': W1_vel,
            'W1_acc': W1_acc}
              
