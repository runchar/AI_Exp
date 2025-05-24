import torch

# save_checkpoint(cfg, model, optimizer, epoch, "last")

def save_checkpoint(cfg, model, optimizer, epoch, name):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    save_path = f"{cfg['output_dir']}/checkpoint_{name}_{epoch}.pth"
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(cfg, model, optimizer, name):
    pass