from dataset import MyDataset, load_data
from utils import get_config  
import argparse
from model import construct_model, construct_loss, construct_optimizer
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm

def test(cfg):
    device = cfg['device']
    model = construct_model(cfg).to(device)

    dataloader = DataLoader(
        MyDataset(cfg, cfg['data']['test_images_path'], None),
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers']
    )