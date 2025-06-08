from torch.optim import AdamW, SGD

def construct_optimizer(cfg, model):
    if cfg['train']['optimizer']['type'] == 'AdamW':
        optimizer = AdamW(
            model.parameters(),
            lr=cfg['train']['optimizer']['params']['lr'],
            weight_decay=cfg['train']['optimizer']['params']['weight_decay']
        )
    elif cfg['train']['optimizer']['type'] == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=cfg['train']['optimizer']['params']['lr'],
            weight_decay=cfg['train']['optimizer']['params']['weight_decay']
        )
    else:
        raise ValueError("Unsupported optimizer type")

    return optimizer