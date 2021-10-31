import torch
from tqdm import tqdm
import numpy as np
import wandb

from src.segocr import SegOCR
from src.dataset import get_dataloaders_and_dict
from config import Config, config


def train_external(config: Config):
    wandb.init(project=config.project_name, name=config.exp_name)

    train_dataloader, val_dataloader, ocr_dict = get_dataloaders_and_dict(
        config.df_path,
        input_size=config.input_size,
        batch_size=config.batch_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = SegOCR(
        input_size=config.input_size,
        output_classes=ocr_dict.count_letters+1,
        rnn_size=config.rnn_size,
        **config.model_kwargs,
    )
    optim = config.optimizer(model.parameters(), **config.optimizer_kwargs)

    loss = config.loss(blank=0, **config.loss_kwargs)
    seg_loss = torch.nn.MSELoss()

    model.to(config.device)
    for epoch in range(config.epoch):
        pbar_train = tqdm(train_dataloader)
        cum_train_losses = []
        for data in pbar_train:
            model.zero_grad()
            optim.zero_grad()
            data['image'] = data['image'].to(config.device)
            data['code'] = data['code'].to(config.device)

            logits, seg_logits = model(data['image'], return_seg=True)
            new_image = seg_logits[:, 1:].max(dim=1)[0].repeat((1, 3, 1, 1))
            new_logits, new_seg_logits = model(new_image, return_seg=True)

            l = loss(logits, data['code'], torch.ones(data['image'].size()[0], device=config.device, dtype=torch.int) * config.rnn_size, data['len'])
            cycle_l = loss(new_logits, data['code'], torch.ones(data['image'].size()[0], device=config.device, dtype=torch.int) * config.rnn_size, data['len'])
            sloss = seg_loss(seg_logits, new_seg_logits)
            (sloss + l + cycle_l).backward()
            optim.step()
            pbar_train.set_postfix({
                'train_loss': l.item(),
                'train_cycle_loss': cycle_l.item(),
                'train_seg_loss': sloss.item(),
            })
            l = sloss + l + cycle_l
            cum_train_losses.append(l.item())

        pbar_val = tqdm(val_dataloader)
        cum_val_losses = []
        cum_val_seg_losses = []
        cum_val_cycle_losses = []
        cum_str_match = []
        for data in pbar_val:
            data['image'] = data['image'].to(config.device)
            data['code'] = data['code'].to(config.device)
            with torch.no_grad():
                logits, seg_logits = model(data['image'], return_seg=True)
                new_image = seg_logits[:, 1:].max(dim=1).repeat((1, 3, 1, 1))
                new_logits, new_seg_logits = model(new_image, return_seg=True)
            l = loss(logits, data['code'],
                     torch.ones(data['image'].size()[0], device=config.device, dtype=torch.int) * config.rnn_size,
                     data['len'])
            cycle_l = loss(new_logits, data['code'],
                         torch.ones(data['image'].size()[0], device=config.device, dtype=torch.int) * config.rnn_size,
                         data['len'])
            sloss = seg_loss(seg_logits, new_seg_logits)
            cur_str_match = []
            for i in range(len(data['text'])):
                timeseries = logits[:, i].argmax(dim=1).detach().cpu().numpy()
                ann_text = data['text'][i]
                pred_text = ocr_dict.timeseries_to_text(timeseries)
                cur_str_match.append(ann_text == pred_text)

            pbar_val.set_postfix({
                'val_loss': l.item(),
                'val_cycle_loss': cycle_l.item(),
                'val_seg_loss': sloss.item(),
                'str_match': np.mean(cur_str_match)
            })
            cum_val_losses.append(l.item())
            cum_val_seg_losses.append(sloss)
            cum_val_cycle_losses.append(cycle_l)
            cum_str_match += cur_str_match

        wandb.log({"train_loss": np.mean(cum_train_losses)})
        wandb.log({"val_loss": np.mean(cum_val_losses)})
        wandb.log({"val_cycle_loss": np.mean(cum_val_cycle_losses)})
        wandb.log({"val_seg_loss": np.mean(cum_val_seg_losses)})
        wandb.log({"str_match": np.mean(cum_str_match)})
        torch.save(model, 'last_checkpoint.pt')


if __name__ == '__main__':
    train_external(config)
