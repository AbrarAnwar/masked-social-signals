import torch

from typing import Any, List
from lightning import LightningModule
import wandb
import os, sys
from utils import MetricTracker

import torch.nn.functional as F
import logging

import numpy as np
import cv2
from copy import deepcopy



class TrainModule(LightningModule):

    def __init__(self, model, optimizer, criterion, metric_ftns, config, lr_scheduler=None):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._logger = logging.getLogger(__name__)

        self.config = config
        
        self.metric_ftns = metric_ftns


        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = MetricTracker(['train/batch/loss'] \
                    + ['train/batch/'+k for k,v in metric_ftns.items()], \
                    log_data=config['train']['log'], mode='train')
                    
        self.valid_metrics = MetricTracker(['val/batch/loss'] \
                    + ['val/batch/'+k for k,v in metric_ftns.items()], \
                    log_data=config['train']['log'], mode='val')


        self.test_metrics = MetricTracker(['test/batch/loss'] \
                    + ['test/batch/'+k for k,v in metric_ftns.items()], \
                    log_data=config['train']['log'], mode='test')





        self.last_metrics = []
        self.last_batch = []
        self.last = False


    #################### GENERAL MODEL ####################


    def forward(self, batch):

        # call forward
        return self.model(batch)
    

    def model_step(self, batch: Any):
        # return loss, pred_actions, gt_actions
        # TODO: take the batch and segment it into 6 parts. maybe in future look into segmenting into 12 parts

        # for a transformer, each segment is a timestep

        # pass it into the model

        # predict the objective we want
        # self.cfg['train']['objectives']

        pass



    #################### TRAIN ####################
    def on_train_start(self):
        self.train_metrics.reset()

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred, target = self.model_step(batch)

        # update and log metrics
        step_number = self.current_epoch *len(self.trainer.train_dataloader) + batch_idx

        self.train_metrics.update('train/batch/loss', loss.item(), step_number)
        for k,met in self.metric_ftns.items():
            self.train_metrics.update('train/batch/'+k, met(pred.cpu().detach(), target.cpu()), step_number)
        return {"loss": loss}

    def on_train_epoch_end(self, outputs):
        log = self.train_metrics.result(self.current_epoch)

        logger_string = ''
        logger_string += "[Epoch {}/{}] \t".format(self.current_epoch, self.config['train']['epochs'])
        logger_string += '\t'.join(['{}: {:0.5f}'.format(k.split('/')[-1], v)  for k,v in log.items()])
        logger_string += '\t lr: {:0.5f}'.format(self.optimizer.param_groups[-1]['lr'])
        self._logger.info(logger_string)

    #################### VALIDATE ####################
    def on_validation_start(self):
        self.valid_metrics.reset()
        self.deca_valid_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred, target = self.model_step(batch)   

        step_number = self.current_epoch * len(self.trainer.val_dataloaders[0]) + batch_idx


        # update and log metrics
        self.valid_metrics.update('val/batch/loss', loss.item(), step_number)
        for k,met in self.metric_ftns.items():
            self.valid_metrics.update('val/batch/'+k, met(pred.cpu().detach(), target.cpu()), step_number)
      
        return {"loss": loss.item()} 
            

    def on_validation_epoch_end(self):
        log = self.valid_metrics.result(self.current_epoch)
        
        print("Setting last to True")
        self.last = True

        logger_string = ''
        logger_string += "[Validate] \t"
        logger_string += '\t'.join(['{}: {:0.5f}'.format(k.split('/')[-1], v)  for k,v in log.items()])
        self._logger.info(logger_string)

        self.log_dict(log)

        return log

    #################### TESTING ####################

    def on_train_start(self):
        self.test_metrics.reset()
    
    def test_step(self, batch: Any, batch_idx: int):

        loss, pred, target = self.model_step(batch)   

        step_number = self.current_epoch * len(self._trainer.test_dataloaders[0]) + batch_idx

        # update and log metrics
        self.test_metrics.update('test/batch/loss', loss.item(), step_number)
        # import pdb; pdb.set_trace()
        for k,met in self.metric_ftns.items():
            self.test_metrics.update('test/batch/'+k, met(pred.cpu().detach(), target.cpu()), step_number)

        # Rajat ToDo: Add deca test metrics
        if self.config['train']['compute_deca_metrics']:
            deca_loss = self.compute_deca_val_metrics(batch, step_number)
            # return deca_loss

        return {"loss": loss.item()}

    def test_epoch_end(self, outputs: List[Any]):
        log = self.test_metrics.result(self.current_epoch)

        if self.config['train']['compute_deca_metrics']:
            deca_log = self.deca_valid_metrics.result(self.current_epoch)
            # log = deca_log

        logger_string = ''
        logger_string += "[Test] \t"
        logger_string += '\t'.join(['{}: {:0.5f}'.format(k.split('/')[-1], v)  for k,v in log.items()])
        self._logger.info(logger_string)

        self.log_dict(log)

        return log

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}

