import torch.nn as nn
import torch
import torch.distributed as dist

def weight_kdloss(x,y,weight):
    kdloss = torch.nn.KLDivLoss()
    n=x.shape[0]
    ret = 0.
    for i in range(n):
        ret += kdloss(x[i],y[i]) * weight[i]
    return ret / n
def weight_MSE(x,y,weight):
    mseloss = torch.nn.MSELoss()
    n=x.shape[0]
    ret = 0.
    for i in range(n):
        ret += mseloss(x[i],y[i]) * weight[i]
    return ret / n

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    epochs = 10.
    lr_warmup = [i / epochs * args.lr for i in range(1, int(epochs +1))]
    if epoch < epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_warmup[epoch]
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def my_KLDivLoss(x, y):
    loss_func = nn.KLDivLoss(reduction='sum')
    y2 = y + 1e-8
    n = x.shape[0]
    loss = loss_func(x, y2) / n
    return loss

# not implemented

def my_sym_klLoss(x,y):
    loss_func = my_KLDivLoss
    
    x_dis = torch.exp(x)
    y_dis = y + 1e-8
    
    loss = 0.5 * loss_func(x, y_dis) + 0.5 * loss_func(y_dis.log(), x_dis + 1e-8)
    return loss