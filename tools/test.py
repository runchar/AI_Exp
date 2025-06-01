from dataset import MyDataset, load_data
from utils import get_config  
import argparse
from model import construct_model, construct_loss, construct_optimizer
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import torch

def test(cfg):
    device = cfg['device']
    model = construct_model(cfg).to(device)

    dataloader = DataLoader(
        MyDataset(cfg, cfg['data']['test_images_path'], None),
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers']
    )

    checkpoint = torch.load(cfg['test']['checkpoint_dir'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            print(f'Output shape: {outputs.shape}')
            print(f'Predictions: {preds}')
    preds = np.array(preds)
    result_file_path = f"{cfg['output_dir']}/final_y.npy"
    np.save(result_file_path, preds)

