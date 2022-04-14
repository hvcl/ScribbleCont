# Public
import os
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Pseudo label generation
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensorV2

# Settings
from scripts.utils import init_logger
from scripts.tb_utils import init_tb_logger
from scripts.metric import Evaluator, AverageMeter
from scripts.optimizer import RAdam

# Contrastive loss
from scripts.contrastive_loss import contrastive_loss1x1, contrastive_loss4x4


class Learner:
    def __init__(self, model, projection_head_1x1, projection_head_4x4, train_loader, valid_loader, cfg):
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model.to(self.cfg.device)
        self.projection_head_1x1 = projection_head_1x1.to(self.cfg.device)
        self.projection_head_4x4 = projection_head_4x4.to(self.cfg.device)
        
        self.logger = init_logger(self.cfg.log_dir, 'train_main.log')
        self.tb_logger = init_tb_logger(self.cfg.log_dir, 'train_main')
        self.log('\n'.join([f"{k} = {v}" for k, v in self.cfg.__dict__.items()]))

        self.summary_loss = AverageMeter()
        self.evaluator = Evaluator()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.cfg.ignore_index)
        self.u_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.cfg.ignore_index)

        train_params = [
            {'params': getattr(model, 'encoder').parameters(), 'lr': self.cfg.lr},
            {'params': getattr(model, 'decoder').parameters(), 'lr': self.cfg.lr * 10},
            {'params': getattr(model, 'segmentation_head').parameters(), 'lr': self.cfg.lr},
            {'params': getattr(projection_head_1x1, 'projection_head').parameters(), 'lr':self.cfg.lr},
            {'params': getattr(projection_head_4x4, 'projection_head').parameters(), 'lr':self.cfg.lr}]

        self.optimizer = RAdam(train_params, weight_decay=self.cfg.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=2, T_mult=2, eta_min=1e-6)

        self.epoch = 0
        self.n_ensemble = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.best_score = -np.inf

    def train_one_epoch(self):
        self.model.train()
        self.summary_loss.reset()
        iters = len(self.train_loader)
        for step, (images, scribbles, weights, _, _) in enumerate(self.train_loader):
            self.tb_logger.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'],
                                      iters * self.epoch + step)
            scribbles = scribbles.to(self.cfg.device).long()
            images = images.to(self.cfg.device)
            batch_size = images.shape[0]            
            self.optimizer.zero_grad()

            outputs, decoder_outputs, layerwise_features = self.model(images)
            
            if self.epoch < self.cfg.thr_epoch:
                loss = self.criterion(outputs, scribbles)
            else:
                scr_loss = self.criterion(outputs, scribbles)
                
                scribbles = scribbles.cpu()
                mean = weights[..., 0]
                u_labels = torch.where(((mean < (1 - self.cfg.thr_conf)) |
                                        (mean > self.cfg.thr_conf)) &
                                       (scribbles == self.cfg.ignore_index),
                                       mean.round().long(),
                                       self.cfg.ignore_index * torch.ones_like(scribbles)).to(self.cfg.device)
                u_loss = self.u_criterion(outputs, u_labels)

                cu_labels = torch.where(((mean < (1 - self.cfg.thr_conf)) |
                                        (mean > self.cfg.thr_conf)) &
                                        (scribbles == self.cfg.ignore_index),
                                        mean,
                                        self.cfg.ignore_index * torch.ones_like(scribbles).float()).to(self.cfg.device)
                cu_labels[scribbles==0] = 0.
                cu_labels[scribbles==1] = 1.
                cu_labels = cu_labels.to(self.cfg.device)

                projections_1x1 = self.projection_head_1x1(decoder_outputs)
                projections_4x4 = self.projection_head_4x4(layerwise_features[3])

                c_loss_1x1 = contrastive_loss1x1(scribbles.to(self.cfg.device), projections_1x1, self.cfg.max_nsample, 
                                        self.cfg.temp1x1, cu_labels)
                c_loss_4x4 = contrastive_loss4x4(projections_4x4, self.cfg.max_nsample, 
                                        self.cfg.temp4x4, cu_labels)
                c_loss = self.cfg.lambda1 * c_loss_1x1 + self.cfg.lambda2 * c_loss_4x4
                
                loss = scr_loss + 0.5 * u_loss + c_loss

            loss.backward()
            self.summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()
        return self.summary_loss.avg
    
    def validation(self):
        self.model.eval()
        self.summary_loss.reset()
        self.evaluator.reset()
        for _, images, targets in self.valid_loader:
            with torch.no_grad():
                targets = targets.to(self.cfg.device).long()
                batch_size = images.shape[0]
                images = images.to(self.cfg.device)
                
                outputs, _, _ = self.model(images)
                loss = self.criterion(outputs, targets)

                targets = targets.cpu().numpy()
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.data.cpu().numpy()
                self.evaluator.add_batch(targets, outputs)
                self.summary_loss.update(loss.detach().item(), batch_size)
        return self.summary_loss.avg, self.evaluator.IoU

    def ensemble_prediction(self):
        ds = self.train_loader.dataset
        transforms = Compose([Normalize(), ToTensorV2()])
        for idx, images in tqdm(ds.images.items(), total=len(ds)):
            augmented = transforms(image=images['image'])
            img = augmented['image'].unsqueeze(0).to(self.cfg.device)
            with torch.no_grad():
                pred = torch.nn.functional.softmax(self.model(img)[0], dim=1)
            weight = torch.tensor(images['weight'])
            pred = pred.squeeze(0).cpu()
            x = pred[1]
            weight[...,0] = self.cfg.alpha * x + (1-self.cfg.alpha) * weight[...,0]
            self.train_loader.dataset.images[idx]['weight'] = weight.numpy()
        self.n_ensemble += 1

    def fit(self, epochs):
        for e in range(epochs):
            t = time.time()
            loss = self.train_one_epoch()

            self.log(f'[Train] \t Epoch: {self.epoch}, loss: {loss:.5f}, time: {(time.time() - t):.2f}')
            self.tb_log(loss, None, 'Train', self.epoch)
            
            t = time.time()
            loss, score = self.validation()

            self.log(f'[Valid] \t Epoch: {self.epoch}, loss: {loss:.5f}, IoU: {score:.4f}, time: {(time.time() - t):.2f}')
            self.tb_log(loss, score, 'Valid', self.epoch)
            self.post_processing(loss, score)

            if (self.epoch + 1) % self.cfg.period_epoch == 0:
                self.log(f'[Ensemble] \t the {self.n_ensemble}th Prediction Ensemble ...')
                self.ensemble_prediction()

            self.epoch += 1
        self.log(f'best epoch: {self.best_epoch}, best loss: {self.best_loss}, best_score: {self.best_score}')

    def post_processing(self, loss, score):
        if loss < self.best_loss:
            self.best_loss = loss

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch

            self.model.eval()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_score': self.best_score,
                'epoch': self.epoch,
            }, f'{os.path.join(self.cfg.log_dir, "best_model.pth")}')

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score = checkpoint['best_score']
        self.epoch = checkpoint['epoch']
        print(f'<model_info>  [best_score] {self.best_score}, [save_epoch] {self.epoch}')

    def log(self, text):
        self.logger.info(text)

    def tb_log(self, loss, IoU, split, step):
        if loss: self.tb_logger.add_scalar(f'{split}/Loss', loss, step)
        if IoU : self.tb_logger.add_scalar(f'{split}/IoU', IoU, step)