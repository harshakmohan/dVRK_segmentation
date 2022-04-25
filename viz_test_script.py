import sys
import torch
import os
from tqdm import tqdm
import torch.optim as optim
# sys.path.insert(1, '/home/imerse/Documents/spring22/CSDL/project/dVRK_segmentation')
from main import parse_command_line
from src.datasets.build_dataloader import UCL_dataloader
from src.models.unet import UNET
from src.run.train_nn import train_fn
from src.run.utils import DiceLoss2D
from src.visualization.viz import viz_val_data, viz_test_data

args = parse_command_line()

# Create dataloader
folder = args['ucl_data_dir']
train_vids = ['Video_01', 'Video_02', 'Video_04', 'Video_05']
test_vids = ['Video_06', 'Video_07']
batch_size = args['batch_size']

train_loader, test_loader = UCL_dataloader(folder, batch_size, train_vids, test_vids)

# Load model with last checkpoint
model = UNET()
model.load_state_dict(torch.load('/home/imerse/Documents/spring22/CSDL/project/dVRK_segmentation/checkpoints/lydia_model_checkpoint.pth'))
model.eval()

viz_val_data(test_loader, model, False)
viz_test_data(test_loader, model, False)
