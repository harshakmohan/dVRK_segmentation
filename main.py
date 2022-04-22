import argparse
import torch
import os
import torch.optim as optim
from src.datasets.build_dataloader import UCL_dataloader
from src.models.unet import UNET
from src.run.train_nn import train_fn
from src.run.utils import DiceLoss2D


def main():
    args = parse_command_line()
    # Testing out some things here for now.
    folder = args['ucl_data_dir']
    train_vids = ['Video_01', 'Video_02', 'Video_04', 'Video_05']
    test_vids = ['Video_06', 'Video_07']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    lr = args['learning_rate']
    model = UNET()

    train_loader, test_loader = UCL_dataloader(folder, batch_size, train_vids, test_vids)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = DiceLoss2D()
    train_fn(train_loader, model, optimizer, loss, num_epochs)


def parse_command_line():
    # TODO: Define all required command line arguments here. This will serve as entry point to the code.

    parser = argparse.ArgumentParser(description='Entry point for robot tool segmentation.')

    parser.add_argument('--run_mode',
                        action='store',
                        default='train',
                        type=str,
                        required=False,
                        choices=['train', 'predict'],
                        metavar='train/predict',
                        help='Specify "train" or "predict".'
                        )

    parser.add_argument('--model',
                        action='store',
                        default='UNET',
                        type=str,
                        required=False,
                        choices=['UNET'],
                        metavar='model',
                        help='If running in train mode, specify which architecture to train on.'
                        )

    current_dir = os.path.abspath(os.getcwd())
    ucl_data_dir = os.path.join(current_dir, 'data/ucl_dataset')
    endovis_data_dir = os.path.join(current_dir, 'data/endovis_dataset')

    parser.add_argument('--ucl_data_dir',
                        action='store',
                        default=ucl_data_dir,  # TODO: Put in default relative path to ucl data folders
                        type=str,
                        required=False,
                        metavar='data',
                        help='Specify the location of the UCL data to be processed.'
                        )

    parser.add_argument('--endovis_data_dir',
                        action='store',
                        default=endovis_data_dir,  # TODO: Put in default relative path to endovis data folders
                        type=str,
                        required=False,
                        metavar='data',
                        help='Specify the location of the endovis data to be processed.'
                        )

    parser.add_argument('--output_checkpoint',
                        action='store',
                        default = '/model_checkpoints/',
                        type=str,
                        required=False,
                        metavar='output_dir',
                        help='Specify the location where output model checkpoint files should be written.'
                        )

    parser.add_argument('--checkpoint_name',
                        action='store',
                        default = 'model_ckpt1',
                        type=str,
                        required=False,
                        metavar='out',
                        help='Specify model checkpoint name'
                        )

    parser.add_argument('--learning_rate',
                        action='store',
                        default = 0.001,
                        type=float,
                        required=False,
                        metavar='learning rate',
                        help='Specify learning rate'
                        )

    parser.add_argument('--device',
                        action='store',
                        default = "cuda:0" if torch.cuda.is_available() else "cpu",
                        type=str,
                        required=False,
                        metavar='compute device',
                        help='Specify compute device'
                        )

    parser.add_argument('--batch_size',
                        action='store',
                        default = 4,
                        type=float,
                        required=False,
                        metavar='batch size',
                        help='Specify batch size'
                        )

    parser.add_argument('--num_epochs',
                        action='store',
                        default = 5,
                        type=float,
                        required=False,
                        metavar='num epochs',
                        help='Specify num epochs'
                        )

    parser.add_argument('--img_width',
                        action='store',
                        default = 538,
                        type=int,
                        required=False,
                        metavar='image width',
                        help='Specify width to crop image to from center'
                        )

    parser.add_argument('--img_height',
                        action='store',
                        default = 701,
                        type=int,
                        required=False,
                        metavar='num epochs',
                        help='Specify height to crop image to from center'
                        )

    parser.add_argument('--ucl_all', # TODO: This arg will indicate if user wants to use specific video folders or if they just wanna use all of them
                        action='store',
                        default = False, # TODO: True if they want to use all of the UCL data. False if they want to use a portion of it.
                        type=bool,
                        required=False,
                        metavar='T/F select all of UCL data or not',
                        help='T/F select all of UCL data or not'
                        )



    return vars(parser.parse_args())


if __name__ == '__main__':
    main()
