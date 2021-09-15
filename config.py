from dataclasses import dataclass
import typing as tp
import torch

from src.segocr import SegOCR


@dataclass
class Config:
    device: str
    num_workers: int
    random_state: int

    df_path: str
    test_size: float

    batch_size: int
    epoch: int

    input_size: tp.Tuple[int, int]
    rnn_size: int

    model: tp.Callable
    model_kwargs: tp.Mapping

    optimizer: tp.Callable
    optimizer_kwargs: tp.Mapping

    loss: tp.Callable
    loss_kwargs: tp.Mapping

    wandb_log: bool


config = Config(
    device='cuda',
    num_workers=0,
    random_state=42,

    df_path='df_path',
    test_size=0.15,
    batch_size=8,
    epoch=100,

    input_size=(64, 512),
    rnn_size=64,

    model=SegOCR,
    model_kwargs={
        'input_size': (64, 512),
    },

    wandb_log=True,
)
