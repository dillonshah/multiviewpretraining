import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchsampler import ImbalancedDatasetSampler
from torchmetrics.functional import auroc
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch.utils.checkpoint as checkpoint

import pandas as pd
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import random
import os

from sklearn.model_selection import train_test_split
from skimage.io import imread

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

val_percent = 0.1
batch_size = 5
epochs = 10                 
num_workers = 4

num_devices = 1

class MammoData(pl.LightningDataModule):
    def __init__(self, val_percent, batch_size, num_workers):
        super().__init__()
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df = self.restructure_data(pd.read_csv("/vol/biomedic3/data/EMBED/tables/merged_df.csv",
         low_memory=False))

        # _, self.df = train_test_split(self.df, test_size=0.2, random_state=42, stratify=self.df['asses'])
        trainval_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42, stratify=self.df['asses'])
        self.train_df, self.val_df = train_test_split(trainval_df, test_size=self.val_percent, random_state=42, stratify=trainval_df['asses'])

        self.train_set = MultiViewDataset(self.train_df)
        self.val_set = MultiViewDataset(self.val_df)
        self.test_set = MultiViewDataset(self.test_df)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, sampler=ImbalancedDatasetSampler(self.train_set), num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    # Creates a new df with each row containing both MLO and CC views of an single side from an individual examination.
    def restructure_data(self, df_in):
        print("Restructuring data...")
        # Keep only relevant rows
        df_in = df_in[['acc_anon','side', 'ViewPosition', 'path_1024png', 'asses']]
        df_in = df_in.dropna()
        # Maintain only MLO and CC views, ignore c-views and other views present in the dataset
        df_in = df_in.loc[(df_in['ViewPosition']=='MLO') | (df_in['ViewPosition']=='CC')]

        result_dict = {}
        views = []
        for _, row in df_in.iterrows():
            side = row['side']
            sideid_anon = str(row['acc_anon']) + side
            view_type = row['ViewPosition']
            asses = row['asses']
            path = row['path_1024png']
            if sideid_anon not in result_dict:
                result_dict[sideid_anon] = {'side': side, 'asses':asses, 'MLO_path': None, 'CC_path': None}
            result_dict[sideid_anon][view_type + '_path'] = path

        # Convert the dictionary into a DataFrame
        return pd.DataFrame.from_dict(result_dict, orient='index').dropna()

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.samples = []
        self.df = df
        for idx, _ in tqdm(self.df.iterrows(), desc="Loading Data", total=len(self.df)):
            sample = {'side': self.df.loc[idx, 'side'], 'MLO_path': self.df.loc[idx, 'MLO_path'],\
             'CC_path': self.df.loc[idx, 'CC_path'], 'asses': self.df.loc[idx, 'asses']}
            self.samples.append(sample)
        
    def __getitem__(self, item):
        sample = self.get_sample(item)

        source = torch.from_numpy(sample['source_view']).unsqueeze(0)
        target = torch.from_numpy(sample['target_view']).unsqueeze(0)

        source = source.repeat(3,1,1)

        # Pad to square image
        pad_height = 1024 - 768 
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        source = torch.nn.functional.pad(source, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)
        target = torch.nn.functional.pad(target, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)

        return {'source':source, 'target':target, 'asses':sample['asses']}

    def get_sample(self, item):
        sample = self.samples[item]
        source_view_path =...
        target_view_path =...

        source_view_path = sample['MLO_path']
        target_view_path = sample['CC_path']

        source_view = imread(source_view_path).astype(np.float32) / 255.0
        target_view = imread(target_view_path).astype(np.float32) / 255.0 

        return {'source_view': source_view, 'target_view': target_view, 'asses':sample['asses']}
    
    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.df['asses'].to_list()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(x)

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        # Remove fc and avgpool layers to preserve spatial information
        self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        # Reshape and infer batch size
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)
        return x

class MammogramModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.model = EncoderDecoder() 
        self.lr = learning_rate
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reconstruction_loss(self, output, target):
        l1_loss = torch.nn.functional.smooth_l1_loss(output, target)
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        return l1_loss/20 + ms_ssim_loss

    def log_images(self, source_image, target_image, output):
        # Combine images into grids for logging
        target_grid = torchvision.utils.make_grid(target_image[0:4, ...], nrow=2, normalize=True)
        output_grid = torchvision.utils.make_grid(output[0:4, ...], nrow=2, normalize=True)

        # Log images to TensorBoard
        self.logger.experiment.add_image('Target Images', target_grid)
        self.logger.experiment.add_image('Generated Images', output_grid)

    def training_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']

        output = self(source_image)  
        loss = self.reconstruction_loss(output, target_image)
        if batch_idx % 5 == 0: 
            self.log_images(source_image, target_image, output)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']

        output = self(source_image)  
        loss = self.reconstruction_loss(output, target_image)

        self.log('val_loss', loss.mean(), batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']

        output = self(source_image) 
        loss = self.reconstruction_loss(output, target_image)

        self.log('test_loss', loss.mean(), batch_size=batch_size)
        
def main():
    data_module = MammoData(val_percent=val_percent, batch_size=batch_size, num_workers=num_workers)
    model = MammogramModel()

    output_base = 'output'
    output_name = 'encoder-decoder18'
    output_dir = os.path.join(output_base, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='mammogram-reconstruction-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=num_devices,
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(output_base, name=output_name),
        log_every_n_steps=5,
    )
    trainer.fit(model, data_module)

    trainer.test(model, datamodule=data_module) 

    # Save the final pre-trained model
    torch.save(model.model.state_dict(), 'pretrained.pth')

if __name__ == '__main__':
    main()
