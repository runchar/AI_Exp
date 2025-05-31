from dataset import MyDataset, load_data
from utils import get_config  
import argparse
from model import construct_model, construct_loss, construct_optimizer
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

def eval_epoch(cfg ,model, dataloader):
    model.eval()
    device = cfg['device']
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += sum(predicted == labels).item()
            total_samples += labels.size(0)

    acc = total_correct / total_samples
    return acc

def train_epoch(cfg ,model, dataloader, optimizer, loss_fn):
    model.train()
    device = cfg['device']
    total_loss = 0

    # for images, labels in tqdm(dataloader, desc="train", leave=False):
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train(cfg):
    device = cfg['device']
    model = construct_model(cfg).to(device)
    optimizer = construct_optimizer(cfg,model)
    loss_fn = construct_loss(cfg)
    start_epoch = 0

    eval_results = []

    if cfg['train']['checkpoint_dir']:
        checkpoint = torch.load(cfg['train']['checkpoint_dir'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1


    train_dataloader = DataLoader(
        MyDataset(cfg, cfg['data']['train_images_path'], cfg['data']['train_labels_path']),
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        MyDataset(cfg, cfg['data']['val_images_path'], cfg['data']['val_labels_path']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
    )

    best_acc = 0.0
    acc = 0.0

    for epoch in range(start_epoch, cfg['train']['epochs']):
        train_loss = train_epoch(cfg, model, train_dataloader, optimizer, loss_fn)


        if (epoch + 1) % cfg['train']['save_interval'] == 0:
            save_checkpoint(cfg, model, optimizer, epoch, "last")

        if (epoch + 1) % cfg['train']['val_interval'] == 0:
            acc = eval_epoch(cfg, model, eval_dataloader)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(cfg, model, optimizer, epoch, "best")
                print(f"save the best model")
            eval_results.append(acc)
                
        print(f"Epoch [{epoch + 1}/{cfg['train']['epochs']}] train loss: {train_loss:.4f}  best acc: {best_acc:.4f}  current acc: {acc:.4f}")

    pd.DataFrame(eval_results, columns=['accuracy']).to_csv(cfg['output_dir'], index=False)
