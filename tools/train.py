from dataset import MyDataset, load_data
from utils import get_config  
import argparse
from model import construct_model, construct_loss, construct_optimizer
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm

def eval_epoch(cfg ,model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="eval", leave=False):
            images = images.to(model.device)
            labels = labels.to(model.device)
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

    for images, labels in tqdm(dataloader, desc="train", leave=False):
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

    train_dataloader = DataLoader(
        MyDataset(cfg, cfg['data']['train_images_path'], cfg['data']['train_labels_path']),
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers']
    )

    eval_dataloader = DataLoader(
        MyDataset(cfg, cfg['data']['val_images_path'], cfg['data']['val_labels_path']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers']
    )

    best_acc = 0.0

    for epoch in range(cfg['train']['epochs']):
        train_loss = train_epoch(cfg, model, train_dataloader, optimizer, loss_fn)


        if (epoch + 1) % cfg['train']['save_interval'] == 0:
            save_checkpoint(cfg, model, optimizer, epoch, "last")

        if (epoch + 1) % cfg['train']['eval_interval'] == 0:
            acc = eval_epoch(cfg, model, eval_dataloader)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(cfg, model, optimizer, epoch, "best")
                print(f"save the best model")
                
        print(f"Epoch [{epoch + 1}/{cfg['train']['epochs']}] train loss: {train_loss:.4f}  best acc: {best_acc:.4f}  current acc: {acc:.4f}")
