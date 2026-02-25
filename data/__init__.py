
from torch.utils.data import DataLoader

from data.datagen.dataset import SRDataset, ErrorDataset, CRCDataset

def build_loaders(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            dataset (dict): Dataset type and image shape configuration.
    """

    # Set dataset type depending on the task
    if opt['task']['type'] == 'SR':
        Dataset = SRDataset
    elif opt['task']['type'] == 'PredictError':
        Dataset = ErrorDataset
    elif opt['task']['type'] == 'CRC':
        Dataset = CRCDataset
        if "splits" in opt['dataset'].keys():
            if "train" in opt['dataset']['splits'].keys():
                del opt['dataset']['splits']['train']
    else:
        Dataset = SRDataset

    # Init datasets and loaders
    loaders = {}
    if "splits" in opt['dataset'].keys():
        for iSplit in opt['dataset']['splits'].keys():
            dataset = Dataset(opt, split=opt['dataset']['splits'][iSplit])
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            loaders[iSplit] = loader
    else:
        dataset = Dataset(opt, split=None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        loaders["test"] = loader
    
    return loaders


