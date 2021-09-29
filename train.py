import torch

from src.segocr import SegOCR
from src.dataset import get_dataloaders_and_dict
from config import Config, config
from tqdm import tqdm


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

    loss = config.loss()#(blank=0, **config.loss_kwargs)

    model.to(config.device)
    for epoch in range(config.epoch):
        pbar = tqdm(train_dataloader)
        for data in pbar:
            model.zero_grad()
            optim.zero_grad()
            data['image'].to(config.device)
            data['code'].to(config.device)
            logits = model(data['image'])
            print(loss, type(loss))
            print(logits.shape, type(logits))
            print(data.keys())
            l = loss(logits, data['code'], torch.ones(data['image'].size[0], device=config.device) * config.rnn_size, data['len'])
            l.backward()
            optim.step()
            pbar.set_postfix({
                'loss': loss,
            })


if __name__ == '__main__':
    train(config)
