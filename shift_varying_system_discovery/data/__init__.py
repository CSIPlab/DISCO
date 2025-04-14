from .div_loader import get_dataloader as get_div_loader, DIV2KDataset
from .sidd_loader import get_dataloader as get_sidd_loader
from .imagenet import ImagenetDataset
from .ffhq import FFHQDataset
from .tmt_data import get_tmt_loader

def get_dataloader(data_set = "div2k", *args, **kwargs):
    if data_set == "div2k":
        return get_div_loader(*args, **kwargs)
    
    elif data_set == 'sidd':
        return get_sidd_loader(*args, **kwargs)
    
    elif data_set  == 'tmt':
        return get_tmt_loader(*args, **kwargs)
    
    raise ValueError(f"Unknown dataset: {data_set}")