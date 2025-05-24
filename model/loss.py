from torch.nn import CrossEntropyLoss

def construct_loss(cfg):
    if cfg['train']['loss']['type'] == 'CE':
        loss_fn = CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss type")
    
    return loss_fn