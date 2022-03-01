import os
import torch
import random
import logging
import warnings
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from apex import amp
from args import get_parser
from models.dbn import DBN
from models.sfcn import SFCN
from models.vgg import vgg16_bn
from models.densenet import densenet121
from models.resnet import resnet18, resnet50

from torch.utils.data import DataLoader
from apex.parallel import DistributedDataParallel
from dataset.data import data_prefetcher, AllData
from utils.utils import *

def initialize():
    # get args
    args = get_parser()

    # warnings
    warnings.filterwarnings("ignore")

    # logger
    logger = logging.getLogger(__name__)

    # set seed
    seed = int(1111)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # initialize logger
    logger.setLevel(level = logging.INFO)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    handler = logging.FileHandler("logs/%s.txt" % args.env_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return args, logger

def main():
    config, logger = initialize()
    config.nprocs = torch.cuda.device_count()
    main_worker(config, logger)

def main_worker(config, logger):
    model_names = ["resnet18", "resnet50", "vgg", "dense121", "sfcn", "dbn"]
    models = [resnet18, resnet50,vgg16_bn, densenet121, SFCN, DBN]

    best_acc1 = 99.0

    dist.init_process_group(backend='nccl')

    # create model
    model = models[model_names.index(config.arch)](output_dim=88)
    T_model = None

    torch.cuda.set_device(config.local_rank)
    model.cuda()

    # load teacher model checkpoint
    if config.teacher_path is not None:
        T_model = models[model_names.index(config.arch)](output_dim=88)
        checkpoint = torch.load(config.teacher_path) 
        T_model.load_state_dict(checkpoint['state_dict'])
        T_model.cuda()

    config.batch_size = int(config.batch_size / config.nprocs)
    optimizer = torch.optim.Adam(model.parameters(),lr = config.lr,weight_decay = 0.00005)
    model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level)
    model = DistributedDataParallel(model)
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData(config.data, train = True)
    val_data = AllData(config.data, train = False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=4,pin_memory = True, sampler = val_sampler)


    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)
        train(train_loader, model, T_model, optimizer, epoch, config, logger)
            
        mae = validate(val_loader, model, config, logger)

        is_best = mae < best_acc1
        best_acc1 = min(mae, best_acc1)
        if not os.path.exists("./checkpoints/%s" % config.env_name):
            try:
                os.makedirs("./checkpoints/%s" % config.env_name)
            except:
                pass # multiple processors bug

        if is_best and config.local_rank == 0:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                    'amp': amp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            torch.save(state, './checkpoints/%s/%s_epoch_%s_%s' % (config.env_name, config.env_name, epoch, best_acc1))

def train(train_loader, model, T_model , optimizer, epoch, config, logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae1', ':6.2f')

    progress = ProgressMeter(len(train_loader), [losses, loss_mae],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)

    model.train()
    if T_model is not None:
        T_model.eval()

    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    while images is not None:
        
        S_out, S_out_p, S_rep = model(images)

        if T_model is not None:
            T_out, T_out_p, T_rep = T_model(images)
            
            S_pred,T_pred = torch.sum(torch.exp(S_out) * bc, dim = 1),torch.sum(torch.exp(T_out) * bc, dim = 1)
            
            weight = abs(T_pred - target) / 5; weight[weight >1] = 1
            weight = 1 - weight

            S_score, T_score = F.log_softmax(S_out_p/config.T,dim=1), F.softmax(T_out_p/config.T,dim=1)

            loss = my_KLDivLoss(S_out, yy)
            losskd = weight_kdloss(S_score, T_score, weight)*config.T*config.T
            lossrep = weight_MSE(S_rep, T_rep, weight)*config.T*config.T
        
            loss = loss + losskd * config.alpha + config.beta * lossrep

        else:
            S_pred = torch.sum(torch.exp(S_out) * bc, dim = 1)
            loss = my_KLDivLoss(S_out, yy)
        mae = torch.nn.L1Loss()(S_pred, target)

        torch.distributed.barrier() 

        reduced_loss = reduce_mean(loss, config.nprocs)
        reduced_mae = reduce_mean(mae, config.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        loss_mae.update(reduced_mae.item(), images.size(0))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            progress.display(i)

        i += 1

        images, target, yy, bc,indices = prefetcher.next()

    logger.info("[train mae]: %.4f" % float(loss_mae.avg))


def validate(val_loader, model, config, logger):

    loss_mae = AverageMeter('mae1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_mae], prefix='Test: ', logger = logger)
    model.eval()

    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target, _, bc, _ = prefetcher.next()
        i = 0
        while images is not None:

            out,_,_ = model(images)

            prob = torch.exp(out)
            pred = torch.sum(prob * bc, dim = 1)
            mae = torch.nn.L1Loss()(pred, target) 
            
            torch.distributed.barrier()
            reduced_mae = reduce_mean(mae, config.nprocs)
            loss_mae.update(reduced_mae.item(), images.size(0))

            if i % config.print_freq == 0:
                progress.display(i)

            i += 1

            images, target, _, bc, _ = prefetcher.next()

        logger.info("[val mae]: %.4f" % float(loss_mae.avg))
    return loss_mae.avg


if __name__ == '__main__':
    main()