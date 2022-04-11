from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from components.wheat_dataset import WheatDataset

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

class WheatDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        csv_file: str = "data/train.csv",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        # read csv file
        train = pd.read_csv(self.hparams.csv_file)

        train[['x', 'y', 'w', 'h']] = pd.DataFrame(
            np.stack(train['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(np.float32)

        # precalculate some values
        train['x1'] = train['x'] + train['w']
        train['y1'] = train['y'] + train['h']
        train['area'] = train['w'] * train['h']


        # split training and validation
        train_ids, valid_ids = train_test_split(train['image_id'].unique(), test_size=0.1)

        train_df = train.loc[train['image_id'].isin(train_ids)]
        valid_df = train.loc[train['image_id'].isin(valid_ids)]

        train_img_dir = f'{self.hparams.data_dir}/train'

        # initialize augmentations
        train_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['train']['augs']]
        train_bbox_params = OmegaConf.to_container((cfg['augmentation']['train']['bbox_params']))
        train_augs = A.Compose(train_augs_list, bbox_params=train_bbox_params)

        valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
        valid_bbox_params = OmegaConf.to_container((cfg['augmentation']['valid']['bbox_params']))
        valid_augs = A.Compose(valid_augs_list, bbox_params=valid_bbox_params)

        train_dataset = WheatDataset(train_df,
                                    'train',
                                    train_img_dir,
                                    cfg,
                                    train_augs)

        valid_dataset = WheatDataset(valid_df,
                                    'valid',
                                    train_img_dir,
                                    cfg,
                                    valid_augs)

        return super().prepare_data()