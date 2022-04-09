import os
from omegaconf import OmegaConf, DictConfig
from utils import load_obj, set_seed, flatten_omegaconf

from model import LitWheat, model_selection
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def train(cfg):
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    # preparing model
    model = model_selection(cfg)
    lit_model = LitWheat(hparams=hparams, cfg=cfg, model=model)

    # checkpoint
    early_stopping = pl.callbacks.EarlyStopping(**cfg.callbacks.early_stopping.params)
    model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg.callbacks.model_checkpoint.params)

    # logger
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)

    # trainer
    trainer = pl.Trainer(logger=[tb_logger],
                     callbacks=[early_stopping],
                     checkpoint_callback=model_checkpoint,
                     **cfg.trainer)

    # fit trainer
    trainer.fit(lit_model)

if __name__ == '__main__':

    ROOT = os.getcwd()
    # get config
    config_path = os.path.join(ROOT, 'config/config.yaml')
    cfg = OmegaConf.load(config_path)

    train(cfg)