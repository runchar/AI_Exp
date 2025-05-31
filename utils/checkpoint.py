import torch

# save_checkpoint(cfg, model, optimizer, epoch, "last")

def save_checkpoint(cfg, model, optimizer, epoch, name):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if name == "best":
        save_path = f"{cfg['output_dir']}/checkpoint_best.pth"
    else:
        save_path = f"{cfg['output_dir']}/checkpoint_{name}_{epoch}.pth"
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(cfg, model, optimizer, name):
    checkpoint = torch.load(f"{cfg['checkpoint_dir']}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']