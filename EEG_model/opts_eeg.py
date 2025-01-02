import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--predict', action='store_true', help='If true, predict is performed.')
    parser.set_defaults(predict=False)
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--path_eeg', default="", type=str, help='Path of seed iv')
    parser.add_argument('--path_cached', default="", type=str, help='Path of cached dataset')
    
    args = parser.parse_args()
    
    return args