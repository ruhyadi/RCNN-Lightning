
from utils import load_obj, set_seed, flatten_omegaconf

from model import LitWheat
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def model_selection(cfg):

    model = load_obj(cfg.model.backbone.class_name)
    model = model(**cfg.model.backbone.params)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(cfg.model.head.class_name)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(in_features, cfg.model.head.params.num_classes)

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
                     early_stop_callback=early_stopping,
                     checkpoint_callback=model_checkpoint,
                     **cfg.trainer)

    # fit trainer
    trainer.fit(lit_model)