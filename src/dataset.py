import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from src.ocr_dict import OcrDict


class OcrDataset(Dataset):
    def __init__(self, df, ocr_dict, input_size=(64, 512)):
        self.df = df
        self.input_size = input_size
        self.ocr_dict = ocr_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # read and prepare image
        im = cv2.imread(row['image'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.input_size[1], self.input_size[0]))
        im = ToTensor()(im)

        # read text and convert text to code
        text = row['text']
        code = self.ocr_dict.fill_code(self.ocr_dict.text_to_code(text))
        code = torch.tensor(code)
        return {
            'image': im,
            'code': code,
            'len': len(text),
        }


def get_dataloaders_and_dict(path_to_df, input_size=(64, 512), batch_size=8, test_size=0.15, random_state=42):
    df = pd.read_csv(path_to_df)
    ocr_dict = OcrDict(df['text'].to_list())
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_dataset = OcrDataset(train_df, ocr_dict, input_size=input_size)
    val_dataset = OcrDataset(val_df, ocr_dict, input_size=input_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, ocr_dict
