import argparse

def get_parser():
    model_names = ["resnet18", "resnet50", "vgg", "dense121", "sfcn", "dbn"]

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default='/path/to/data', help='path to dataset')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet18',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--epochs', default=260, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 64), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=3e-4,
                        type=float,
                        metavar='LR',
                        help='learning rate for all GPUs.',
                        dest='lr')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('-t','--teacher_path', default = None, help='teacher model path. Train Student with None.')
    parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--lam', default=0.5, type=float, help='lam value. ')
    
    parser.add_argument('--T', default=1, type=int, help='alpha value. ')
    parser.add_argument('--beta', default=0.8, type=float, help='alpha value. ')
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha value. ')
    parser.add_argument('--scale', default=5, type=float, help='alpha value. ')

    parser.add_argument('--env_name', default = "default", help='name for env')
    parser.add_argument('--opt_level', default = "O1", help='opt level, O1 default')


    args = parser.parse_args()
    args.env_name = args.env_name +  "-model.%s-lam.%s-alpha.%s-beta.%s-scale.%s-T.%s" % \
        (args.arch, args.lam, args.alpha, args.beta, args.scale, args.T)
    return args