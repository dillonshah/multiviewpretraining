import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchsampler import ImbalancedDatasetSampler
from torchmetrics.functional import auroc
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.util import img_as_ubyte
from skimage.io import imread

import pandas as pd
from tqdm import tqdm
from sampler import SamplerFactory
import numpy as np

import cv2
import random
import os
import numbers

test_percent = 0.2
val_percent = 0.1
batch_size = 10
epochs = 15                 
num_workers = 4

num_devices = 2

# Transfrom taken from biomedia's mammo-net.py
class GammaCorrectionTransform:
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')   
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        gamma_factor = None if self.gamma is None else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img

class EMBEDData(pl.LightningDataModule):
    def __init__(self, val_percent, test_percent, batch_size, num_workers):
        super().__init__()
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df = pd.read_csv("/vol/biomedic3/data/EMBED/tables/merged_df.csv",
         low_memory=False)

        # FFDM only
        self.df = self.df[self.df['FinalImageType'] == '2D']

        # Female only
        self.df = self.df[self.df['GENDER_DESC'] == 'Female']

        # Remove unclear breast density cases
        self.df = self.df[self.df['tissueden'].notna()]
        self.df = self.df[self.df['tissueden'] < 5]

        # MLO and CC only        
        self.df = self.df[self.df['ViewPosition'].isin(['MLO','CC'])]

        # Remove spot compression or magnificiation
        self.df = self.df[self.df['spot_mag'].isna()]

        # Restructure data for multi-view task
        self.df = self.restructure_data(self.df)

        self.df['mlo_path'] = self.df.MLO_path.values
        self.df['cc_path'] = self.df.CC_path.values
        self.df['study_id'] = [str(study_id) for study_id in self.df.study_id.values]

        # Making sure images from the same subject are within the same set
        self.df['split'] = 'test'

        unique_study_ids_all = self.df.study_id.unique()
        unique_study_ids_all = shuffle(unique_study_ids_all)
        num_test = (round(len(unique_study_ids_all) * self.test_percent))
        
        dev_sub_id = unique_study_ids_all[num_test:]
        self.df.loc[self.df.study_id.isin(dev_sub_id), 'split'] = 'training'
        
        self.dev_data = self.df[self.df['split'] == 'training']
        self.test_data = self.df[self.df['split'] == 'test']        

        unique_study_ids_dev = self.dev_data.study_id.unique()

        unique_study_ids_dev = shuffle(unique_study_ids_dev)
        num_train = (round(len(unique_study_ids_dev) * (1.0 - self.val_percent)))

        valid_sub_id = unique_study_ids_dev[num_train:]
        
        self.dev_data.loc[self.dev_data.study_id.isin(valid_sub_id), 'split'] = 'validation'
        
        self.train_data = self.dev_data[self.dev_data['split'] == 'training']
        self.val_data = self.dev_data[self.dev_data['split'] == 'validation']

        self.train_set = MultiViewDataset(self.train_data, augment=True)
        self.val_set = MultiViewDataset(self.val_data)
        self.test_set = MultiViewDataset(self.test_data)

        print('samples (train): ',len(self.train_set))
        print('samples (val):   ',len(self.val_set))
        print('samples (test):  ',len(self.test_set))
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
    # Creates a new df with each row containing both MLO and CC views of an single side from an individual examination.
    def restructure_data(self, df_in):
        print("Restructuring data...")
        
        result_dict = {}
        views = []
        for _, row in df_in.iterrows():
            side = row['side']
            sideid_anon = str(row['acc_anon']) + side
            view_type = row['ViewPosition']
            path = row['path_1024png']
            if sideid_anon not in result_dict:
                result_dict[sideid_anon] = {'asses':row['asses'],  'study_id': row['empi_anon'], 'MLO_path': None, 'CC_path': None}
            result_dict[sideid_anon][view_type + '_path'] = path

        # Convert the dictionary into a DataFrame
        return pd.DataFrame.from_dict(result_dict, orient='index').dropna()

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, df, augment = False):
        self.augment = augment
        self.df = df

        # photometric data augmentation
        self.photometric_augment = T.Compose([
            GammaCorrectionTransform(gamma=0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        
        # geometric data augmentation
        self.geometric_augment = T.Compose([
            T.RandomApply(transforms=[T.RandomAffine(degrees=10, scale=(0.9, 1.1))], p=0.5),
        ])

        self.mlo_paths = df.MLO_path.to_numpy()
        self.cc_paths = df.CC_path.to_numpy()


    def preprocess(self, image):
        # breast mask
        image_norm = image - np.min(image)
        image_norm = image_norm / np.max(image_norm)
        thresh = cv2.threshold(img_as_ubyte(image_norm), 5, 255, cv2.THRESH_BINARY)[1]

        # Connected components with stats.
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)

        # Find the largest non background component.
        max_label, _ = max(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
            key=lambda x: x[1],
        )
        mask = output == max_label
        image[mask == 0] = 0
        
        return image
        
    def __getitem__(self, index): 
        # Randomly choose between mlo and cc, and unpack the paths accordingly
        source_path, target_path = random.choice([
            (self.mlo_paths[index], self.cc_paths[index]),
            (self.cc_paths[index], self.mlo_paths[index])
        ])
        
        source = imread(source_path).astype(np.float32) / 65535.0
        target = imread(target_path).astype(np.float32) / 65535.0
        
        source = self.preprocess(source)
        target = self.preprocess(target)
        
        source = torch.from_numpy(source).unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)

        if self.augment:
            source = self.geometric_augment(self.photometric_augment(source))
        
        source = source.repeat(3,1,1)

        # Pad to square image
        pad_height = 1024 - 768 
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        source = torch.nn.functional.pad(source, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)
        target = torch.nn.functional.pad(target, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)

        return {'source':source, 'target':target}
    
    def __len__(self):
        return len(self.df)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=0),
        )
        
    def forward(self, x):
        return self.decoder(x)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(512, 512), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(512, 512), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove fc and avgpool layers to preserve spatial information
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        # Downsample using maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.maxpool(x)
        x = self.decoder(x)
        return x

class MultiViewModel(pl.LightningModule):
    def __init__(self, learning_rate=0.0001, checkpoint=None):
        super().__init__()
        self.model = EncoderDecoder() 

        if checkpoint is not None:
            print(self.model.load_state_dict(state_dict=checkpoint, strict=False))

        self.lr = learning_rate
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.vgg_perceptual = VGGPerceptualLoss() 

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reconstruction_loss(self, output, target, alpha = 0.15, beta = 0.15):
        normalised_output = (output-output.min())/(output.max()-output.min()) 
        l1_loss = torch.nn.functional.l1_loss(normalised_output, target)
        ms_ssim_loss = 1 - self.ms_ssim(normalised_output, target)
        perceptual_loss = self.vgg_perceptual(normalised_output, target)
        return ((1 - alpha - beta) * l1_loss + alpha * ms_ssim_loss + beta * perceptual_loss, \
         (l1_loss, ms_ssim_loss, perceptual_loss))

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
        loss, losses = self.reconstruction_loss(output, target_image)
        if batch_idx % 5 == 0: 
            self.log_images(source_image, target_image, output)
        self.log('train_loss', loss, batch_size=batch_size)
        self.log('Smooth L1 Loss', losses[0], batch_size=batch_size)
        self.log('MS-SSIM Loss', losses[1], batch_size=batch_size)
        self.log('Perceptual Loss', losses[2], batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']

        output = self(source_image)  
        loss, _ = self.reconstruction_loss(output, target_image)

        self.log('val_loss', loss.mean(), batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']

        output = self(source_image) 
        loss, _ = self.reconstruction_loss(output, target_image)

        self.log('test_loss', loss.mean(), batch_size=batch_size)

class MultiViewEmbeddings(MultiViewModel):
    def __init__(self, init=None):
        super().__init__()
        self.embeddings = []
        if init is not None:
            self.model = init.model

    def on_test_start(self):
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity(num_features)
        self.embeddings = []

    def test_step(self, batch, batch_idx):
        emb = self.forward(batch['source'])
        self.embeddings.append(emb.detach().cpu())

    def on_test_epoch_end(self):
        self.embeddings = torch.cat(self.embeddings, dim=0)
        
def main():
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(42, workers=True)
    data_module = EMBEDData(val_percent=val_percent, test_percent=test_percent, batch_size=batch_size, num_workers=num_workers)
    model = MultiViewModel()

    output_base = 'output'
    output_name = 'encoder-decoder18'
    output_dir = os.path.join(output_base, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    torch.save(model.model.state_dict(), 'pretrained.pth')

    print("Saving Embeddings")

    model_modified = MultiViewEmbeddings(init=model)
    trainer.test(model=model_modified, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)
    save_embeddings(model=model_modified, output_fname=os.path.join(output_dir, 'embeddings.csv'))


if __name__ == '__main__':
    main()
