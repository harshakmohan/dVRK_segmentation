import argparse
import torch
import os
import os.path as osp
import torch.optim as optim
import torch.nn as nn
from src.datasets.build_dataloader import UCL_dataloader, Endovis_Binary_dataloader
from src.run.utils import check_accuracy, load_checkpoint
from src.models.unet import UNET
from src.run.train_nn import train_fn
from src.run.predict import save_predictions_as_imgs as save_imgs
from src.run.utils import DiceLoss2D, SoftDiceLoss, init_weights


def main():
    args = parse_command_line()
    cwd = os.getcwd()
    binary_endovis_path = osp.join(osp.join(cwd, 'data'), 'binary_endovis')

    # Unpack args
    folder = args['ucl_data_dir']
    run_mode = args['run_mode']
    checkpoint_name = args['checkpoint_name']

    train_vids = ['Video_01', 'Video_02', 'Video_03', 'Video_04', 'Video_05', 'Video_06', 'Video_07', 'Video_08', 'Video_09', 'Video_10', 'Video_12', 'Video_13']
    test_vids = ['Video_14']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    device = args['device']
    lr = args['learning_rate']

    if run_mode=='train':
        print('Preparing to train!')
        # Prepare data loaders, optimizer, loss fn
        model = UNET().to(device)
        model.apply(init_weights)
        train_loader, test_loader = UCL_dataloader(folder, batch_size, train_vids, test_vids)
        #train_loader = Endovis_Binary_dataloader(binary_endovis_path, batch_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss = DiceLoss2D()
        #loss = nn.BCEWithLogitsLoss()
        #loss = SoftDiceLoss()

        train_fn(train_loader, test_loader, model, optimizer, loss, num_epochs, checkpoint_name)

    elif run_mode=='predict':
        print('Preparing to predict!')
        # TODO: If predict mode selected, then perform inference using a model and selected data then save predictions as images
        # checkpoint = '/home/harsha/PycharmProjects/dVRK_segmentation/checkpoints/model_checkpoint.pth'
        model = UNET().to(device)
        checkpoint = torch.load('/home/harsha/PycharmProjects/dVRK_segmentation/checkpoints/unet_checkpoint_95acc_82ds.tar')['state_dict']
        model.load_state_dict(checkpoint)

        train_loader, test_loader = UCL_dataloader(folder, batch_size, train_vids, test_vids)
        save_imgs(test_loader, model, device)
        # check_accuracy(test_loader, model)


    else:
        raise ValueError('Incorrect run mode specified. Either specify "train" or "predict"...')



def parse_command_line():

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

    parser.add_argument('--loss_fn',
                        action='store',
                        default='BCEWithLogitsLoss',
                        type=str,
                        required=False,
                        choices=['BCEWithLogitsLoss', 'DiceLoss2D', 'SoftDiceLoss'],
                        metavar='loss_fn',
                        help='Specify which loss function to use...'
                        )

    current_dir = os.path.abspath(os.getcwd())
    ucl_data_dir = os.path.join(current_dir, 'data/ucl_dataset')
    endovis_data_dir = os.path.join(current_dir, 'data/binary_endovis')

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
                        default = 'unet_endovis',
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
                        default = 2,
                        type=float,
                        required=False,
                        metavar='batch size',
                        help='Specify batch size'
                        )

    parser.add_argument('--num_epochs',
                        action='store',
                        default = 50,
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

    return vars(parser.parse_args())


if __name__ == '__main__':
    main()