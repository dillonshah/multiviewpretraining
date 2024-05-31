import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from skimage.io import imread
import os
from pretrain import MultiViewModel, EncoderDecoder


class MammoData():
    def __init__(self):
        self.df = self.restructure_data(pd.read_csv("/vol/biomedic3/data/EMBED/tables/merged_df.csv",
            low_memory=False))

    # Creates a new df with each row containing both MLO and CC views of an single side from an individual examination.
    def restructure_data(self, df_in):
        print("Restructuring data...")
        df_in = df_in[['acc_anon','side', 'ViewPosition', 'path_1024png', 'asses']]
        df_in = df_in.dropna()
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
    
    def get_random_sample_pair(self, idx):
        row = self.df.iloc[idx]
        return row['MLO_path'], row['CC_path']


def load_model(model_path, device='cuda', checkpoint=False): 
    if checkpoint:
        mammo_model = MultiViewModel.load_from_checkpoint(model_path)
        model = mammo_model.model
    else:
        model = EncoderDecoder()
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval() 
    return model

def reconstruct_image(model, image_path, device='cuda'):
    source_view = imread(image_path).astype(np.float32) / 255 
    source = torch.from_numpy(source_view).unsqueeze(0).repeat(3,1,1).unsqueeze(0)
    pad_height = 1024 - 768 
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    source = torch.nn.functional.pad(source, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)
    source = source.to(device)
    print(source.shape)
    with torch.no_grad():
        output = model(source)
    output_image = output.squeeze(0).repeat(3,1,1).transpose(0,1).transpose(1,2).cpu().numpy()  
    return normalize(output_image)

def normalize(image):
    min_val = image.min()
    max_val = image.max()
    print(min_val, max_val)
    normalized_image = (image - min_val) / (max_val - min_val) * 255  
    return normalized_image.astype(np.uint8) 

if __name__ == "__main__":
    mammo_data = MammoData()
    model = load_model("/vol/biomedic3/bglocker/ugproj2324/ds1021/multiviewpretraining/pretraining/runs/smoothl1/pretrained.pth", checkpoint=False)
    while True:
        idx = int(input("Enter an index to reconstruct an image: "))
        mlo_path, cc_path = mammo_data.get_random_sample_pair(idx)
        output_image = reconstruct_image(model, mlo_path, device='cuda')
        print("MLO Path: ", mlo_path)
        print("CC Path: ", cc_path)

        output_dir = 'saved_images'
        os.makedirs(output_dir, exist_ok=True)

        image_filename = os.path.basename(cc_path)
        output_filepath = os.path.join(output_dir, f'{idx}_reconstructed_{image_filename}')

        plt.imsave(output_filepath, output_image) 
