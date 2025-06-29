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
import os
from PIL import Image

def eval_epoch(cfg ,model, dataloader):
    model.eval()
    device = cfg['device']
    total_correct = 0
    total_samples = 0

    predicteds = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += sum(predicted == labels).item()
            
            # Save misclassified images and their labels
            misclassified = predicted != labels
            mis_images = images[misclassified].cpu()
            mis_labels = labels[misclassified].cpu().numpy()
            mis_preds = predicted[misclassified].cpu().numpy()
            for idx in range(mis_images.size(0)):
                img = mis_images[idx]
                # If image has 1 channel, squeeze; if 3, permute to HWC
                if img.size(0) == 1:
                    img_to_save = img.squeeze(0).numpy()
                else:
                    img_to_save = img.permute(1, 2, 0).numpy()
                # Normalize to 0-255 and convert to uint8
                img_to_save = (img_to_save - img_to_save.min()) / (img_to_save.max() - img_to_save.min() + 1e-5)
                img_to_save = (img_to_save * 255).astype(np.uint8)
                # Save image with label and predicted in filename
                save_dir = os.path.join(cfg['output_dir'], 'misclassified')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"img_{len(os.listdir(save_dir))}_label_{mis_labels[idx]}_pred_{mis_preds[idx]}.png")
                Image.fromarray(img_to_save).save(save_path)

            predicteds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            total_samples += labels.size(0)

    np.save(cfg['output_dir'] + '/predicted.npy', np.array(predicteds))
    np.save(cfg['output_dir'] + '/labels.npy', np.array(labels_list))

    acc = total_correct / total_samples
    return acc

def train_epoch(cfg ,model, dataloader, optimizer, loss_fn):
    model.train()
    device = cfg['device']
    total_loss = 0

    for images, labels in tqdm(dataloader, desc="train", leave=False):
    # for images, labels in dataloader:
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
