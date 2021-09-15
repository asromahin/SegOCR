from src.segocr import SegOCR
from src.dataset import get_dataloaders_and_dict
from config import Config, config

INPUT_SIZE = (64, 512)
WANDB_LOG = True
PATH_DF = ''


def train(config: Config):
    train_dataloader, val_dataloader, ocr_dict = get_dataloaders_and_dict(
        config.df_path,
        input_size=config.input_size,
    )
    model = SegOCR(
        input_size=config.input_size,
        output_classes=ocr_dict.count_letters+1,
        rnn_size=config.rnn_size,
        **config.model_kwargs,
    )
    optim = config.optimizer(model.parameters(), **config.optimizer_kwargs)

    loss = config.loss(**config.loss_kwargs)

    #for epoch in range(config.epoch):


if __name__ == '__main__':
    train(config)
