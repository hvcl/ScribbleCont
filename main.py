# public
import os
import pandas as pd
import torch
from torch.utils.data  import DataLoader
from scripts.optimizer import RAdam

# settings
from model.model import UnetCustom, projection_head_1x1, projection_head_4x4
from scripts.dataset import get_transforms, dsbTrainDataset, dsbValidDataset
from scripts.learner import Learner
from scripts.utils   import seed_everything


class config:
    seed = 42
    name = 'MoNuSeg' # bright_field, histopathology, fluorescence, MoNuSeg
    device = torch.device('cuda:0')
    scr = 'manual'
    fold = 0
    """ Path """
    data_dir = f'./examples/images/{name}/'
    mask_dir = f'./examples/labels/{name}/full/'
    df_path  = f'./examples/labels/{name}/train.csv'
    scr_dir  = f'./examples/labels/{name}/scribble_{scr}/'
    log_dir  = f'./logs/'
    """ Training """
    n_epochs = 10000
    input_size = 256
    batch_size = 30
    lr = 3e-4
    weight_decay = 5e-5
    num_workers = 8
    ignore_index = 250
    """ Scribble Params """
    thr_epoch = 100
    period_epoch = 5
    if name=='MoNuSeg':
        thr_conf = 0.95
        alpha = 0.1
    else:
        thr_conf = 0.8
        alpha = 0.2
    """ Contrastive Params """
    lambda1 = 0.5
    lambda2 = 10.
    temp1x1 = 0.3
    temp4x4 = 0.1
    max_nsample = 6000


if __name__ == '__main__':
    seed_everything(config.seed)
    
    net = UnetCustom(encoder_name='resnet34' ,encoder_weights='imagenet', decoder_attention_type='scse', activation=None, classes=2)
    projection_head_1x1 = projection_head_1x1()
    projection_head_4x4 = projection_head_4x4()

    df = pd.read_csv(config.df_path)
    train_df = df[df.fold != config.fold].reset_index(drop=True)
    valid_df = df[df.fold == config.fold].reset_index(drop=True)

    ''' Data load '''
    # Data augmentation
    if config.name=='MoNuSeg':
        # MoNuSeg: color jittering
        transforms = get_transforms(config.input_size, jittering=True, need=('train', 'val'))
    else:
        # Ohters: brightness contrast
        transforms = get_transforms(config.input_size, jittering=False, need=('train', 'val'))
    
    train_dataset = dsbTrainDataset(config.data_dir, config.scr_dir, config.mask_dir, train_df, tfms=transforms['train'])
    valid_dataset = dsbValidDataset(config.data_dir, config.mask_dir, valid_df, tfms=transforms['val'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, num_workers=config.num_workers, shuffle=False)
    
    Learner = Learner(net, projection_head_1x1, projection_head_4x4, train_loader, valid_loader, config)
    pretrained_path = os.path.join(config.log_dir, 'best_model.pth')
    if os.path.isfile(pretrained_path):
        Learner.load(pretrained_path)
        Learner.log(f"Checkpoint Loaded: {pretrained_path}")
    Learner.fit(config.n_epochs)