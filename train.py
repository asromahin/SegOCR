from src.segocr import SegOCR
from src.dataset import get_dataloaders_and_dict

INPUT_SIZE = (64, 512)
WANDB_LOG = True
PATH_DF = ''


def train():
    train_dataloader, val_dataloader, ocr_dict = get_dataloaders_and_dict(
        PATH_DF,
        input_size=INPUT_SIZE,
    )
    model = SegOCR(input_size=INPUT_SIZE)


if __name__ == '__main__':
    train()
