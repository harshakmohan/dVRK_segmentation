
def main():
    args = parse_command_line()
    # TODO: Build out rest of main function here.


def parse_command_line():
    # TODO: Define all required command line arguments here. This will serve as entry point to the code.
    # TODO: @Harsha include arguments for all the hyperparameters for training
    import argparse

    parser = argparse.ArgumentParser(description='Entry point for robot tool segmentation.')

    parser.add_argument('--run_mode',
                        action='store',
                        default='train',
                        type=str,
                        required=True,
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

    parser.add_argument('--data_folder',
                        action='store',
                        default='',  # TODO: Put in default relative path to data folders
                        type=str,
                        required=True,
                        metavar='data',
                        help='Specify the location of the data to be processed.'
                        )

    parser.add_argument('--output_checkpoint',
                        action='store',
                        default = '/model_checkpoints/',
                        type=str,
                        required=True,
                        metavar='output_dir',
                        help='Specify the location where output model checkpoint files should be written.'
                        )

    parser.add_argument('--checkpoint_name',
                        action='store',
                        type=str,
                        required=True,
                        metavar='out',
                        help='Specify model checkpoint name'
                        )

    return vars(parser.parse_args())


if __name__ == '__main__':
    main()
