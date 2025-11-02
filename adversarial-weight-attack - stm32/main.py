from __future__ import division
from __future__ import absolute_import

import configs

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth
#from tensorboardX import SummaryWriter
import models
from models.quantization import quan_Conv2d, quan_Conv1d, quan_Linear, quantize

from torch.optim import SGD, lr_scheduler

from attack.BFA import *
import torch.nn.functional as F
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

from training_tools import *

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'mit-bih'],
    help='Choose between Cifar10 and MIT-BIH datasets.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='cnn_quan',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))


parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')


parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')

parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
                 
parser.add_argument('--clipping_coeff',
                    type=float,
                    default=0.1,
                    help='coefficient to control clipping strenght')


parser.add_argument('--randbet',
                    dest='randbet',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')
                    
                    
parser.add_argument('--limit_layer',
                    type=int,
                    default=-1,
                    help='Layen that can be targeted')
                    
parser.add_argument('--randbet_coeff',
                    type=int,
                    default=10,
                    help='Coefficient of ramdomization for randbet')

##########################################################################

opt = parser.parse_args()

args = configs.args(opt)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_cuda:
      torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

    
set_seed(args.manualSeed)


###############################################################################
###############################################################################


def main():
    # Init logger6
    
    
    print(f"------------------------USE_CUDA: {args.use_cuda}")
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'mit-bih':
        print("dataset mi-bit")
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])


    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10

    elif args.dataset == 'mit-bih':
    
        X_train_full = np.load('dataset/X_train.npy')
        y_train_full = np.load('dataset/y_train.npy')
        X_test = np.load('dataset/X_test.npy')
        y_test = np.load('dataset/y_test.npy')
        
        
        le = LabelEncoder()
        le.fit(y_train_full)
        y_train_full = le.transform(y_train_full)
        y_test = le.transform(y_test)
        
        
        scaler = preprocessing.StandardScaler().fit(X_train_full)
        X_train_scaled = scaler.transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)
        
        print("[MIT-BIH] X_train_full: ", X_train_full.shape, 'X_test:', X_test.shape)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_full, test_size=0.30, random_state=42)
        
        print("[MIT-BIH] X_train: ", X_train.shape, 'X_val:', X_val.shape)
        
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)  # Adding channel dimension
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        num_classes = 5
            
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)
        
        
    if args.dataset == 'cifar10':

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.attack_sample_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    elif args.dataset == 'mit-bih':

        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    else:
    
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    verif_cuda(net)

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("Optimizer=SGD")
        optimizer = torch.optim.SGD(all_param, lr=state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

    elif args.optimizer == "Adam":
        print("Optimizer=Adam")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, all_param), lr=state['learning_rate'], weight_decay=state['decay'])


    if args.use_cuda:
        net.cuda()
        criterion.cuda
    else:
        net.cpu()
        
    verif_cuda(net)


    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(
                args.resume,
                weights_only=False,                 # <-- important
                map_location=('cpu' if not args.use_cuda else None)
            )
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)

            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)
        
    weights_exploitation(net, args.save_path, bfa_position=0, info="32_bits")
    
    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net, args.quan_bitwidth)
        
        
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
            print(f"Layer available: {m}")
            print(f"# of weifghts {m.weight.numel()}")
        

    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
                m.__reset_weight__()
                # print(m.weight)

    weights_exploitation(net, args.save_path, bfa_position=0, info="8_bits")

    attacker = BFA(criterion, net, args.k_top)
    net_clean = copy.deepcopy(net)
    # weight_conversion(net)

    if args.enable_bfa:
        print("Layers construction and numerotation")
        for module_idx, (name, module) in enumerate(net.named_modules()):
          print(f"{module_idx} ------- {name} | {module}")
    
        perform_attack(attacker, net, net_clean, train_loader, test_loader, args.n_iter, log, criterion, csv_save_path=args.save_path, random_attack=args.random_bfa)

        return

    if args.evaluate:
        _,_,_, output_summary = validate(test_loader, net, criterion, log, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    

    for epoch in range(args.start_epoch, args.epochs):
        
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        
        #current_learning_rate = scheduler.get_last_lr()
        
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)


        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        
        
        print(args.randbet)
        if args.clipping_coeff != 0 and not args.randbet:
        
            print("Clipping training")
            for m in net.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
                    m.weight.data = torch.clamp(m.weight.data, min = -args.clipping_coeff, max = args.clipping_coeff)
            train_acc, train_los = train_clipping(train_loader, net, criterion, optimizer, epoch, log, clipping_coeff=args.clipping_coeff)
            
        elif args.randbet:
        
            print("RandBet training")
            for m in net.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
                    m.weight.data = torch.clamp(m.weight.data, min = -args.clipping_coeff, max = args.clipping_coeff)
            train_acc, train_los = train_randbet(train_loader, net, criterion, optimizer, epoch, log, clipping_coeff=args.clipping_coeff)        
        else:
            print("Standard training")
            train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log)

        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

    # ============ TensorBoard logging ============#
    log.close()


def perform_attack(attacker, model, model_clean, train_loader, test_loader, N_iter, log, criterion, csv_save_path=None, random_attack=False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146

    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    # evaluate the test accuracy of clean model
    val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model, attacker.criterion, log, summary_output=True)
    tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    tmp_df['BFA iteration'] = 0
    tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_0.csv'.format(args.arch)), index=False)

    print_log('k_top={}'.format(args.k_top), log)
    print_log('Attack_sample={}'.format(data.size()[0]), log)
    end = time.time()
    
    df = pd.DataFrame() #init a empty dataframe for logging
    #df = pd.DataFrame(columns = range(8))
    last_val_acc_top1 = val_acc_top1 
    
    grads_exploitation(data, target, model, criterion, args.save_path, bfa_position=0)
    #weights_exploitation(model, args.save_path, bfa_position=0)
    
    
    for i_iter in range(N_iter):
        print_log('************** ATTACK iteration *****************', log)
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target)
        else:
            attack_log = attacker.random_flip_one_bit(model)
            
        
        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                    log)
            print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        except:
            pass
        
        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)


        # exam the BFA on entire val dataset
        val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model, attacker.criterion, log, summary_output=True)
        tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        tmp_df['BFA iteration'] = i_iter + 1
        tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_{}.csv'.format(args.arch, i_iter + 1)), index=False)
    
        
        # add additional info for logging
        acc_drop = last_val_acc_top1 - val_acc_top1
        last_val_acc_top1 = val_acc_top1
        
        # print(attack_log)
        for i in range(attack_log.__len__()):
            attack_log[i].append(val_acc_top1)
            attack_log[i].append(acc_drop)
        #print(attack_log[0])
        #df = df.append(attack_log, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([attack_log[0]])], ignore_index=True)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log('iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(iter_time=iter_time), log)
        end = time.time()
        
        # Stop the attack if the accuracy is below the configured break_acc.
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'mit-bih':
            break_acc = 21.0
        if val_acc_top1 <= break_acc:
            break
        
    # attack profile
    #column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx', 'weight before attack', 'weight after attack', 'validation accuracy', 'accuracy drop']
    #df.columns = column_list
    #df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None, header=False)

    return


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    
    verif_cuda(model)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(non_blocking=True)  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg
    
    
def train_clipping(train_loader, model, criterion, optimizer, epoch, log, clipping_coeff=0.1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    verif_cuda(model)

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
    
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(non_blocking=True)  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # compute output
        
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
                m.weight.data = torch.clamp(m.weight.data, min = -clipping_coeff, max = clipping_coeff)
        
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    
    

    
    return top1.avg, losses.avg


def train_randbet(train_loader, model, criterion, optimizer, epoch, log, clipping_coeff=0.1, limit_layer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    verif_cuda(model)
    

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        
        
        data_time.update(time.time() - end)
        
        for module_idx, (name, m) in enumerate(model.named_modules()):
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, quan_Conv1d):
                #print(f"max:{torch.max(m.weight)}")
                #print(f"min:{torch.min(m.weight)}")
                with torch.no_grad():
                    m.weight.data = torch.clamp(m.weight, min = -clipping_coeff, max  = clipping_coeff)
                #print(f"max:{torch.max(m.weight)}")
                #print(f"min:{torch.min(m.weight)}")     
        
        

        if args.use_cuda:
            target = target.cuda(non_blocking=True)  # the copy will be asynchronous with respect to the host.
            input = input.cuda()
        
        model.eval()
        model_clean = copy.deepcopy(model).cpu()
        model_clean.train()
        model.train()
        
        optimizer.zero_grad()
        output = model(input)        
        loss = criterion(output, target)

        #Here code pour faire random attaque
        model = apply_global_bit_flips(model, args.randbet_coeff, args.limit_layer)
        
        if args.use_cuda:
          model.cuda()
        
        
        utput = model(input)
        loss_faulted = criterion(output, target)
        
        (loss+loss_faulted).backward()
        
        
        transfer_weights(model_clean, model)
        
        
        model = divide_gradients_by_two(model)
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            
            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)
                
                #for j in range(len(output_summary)):
                #    print(i, len(output_summary), np.shape(output_summary[j]))

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write('{epoch}       {train}    {test}\n'.format(**recorder))
        
        
        
def verif_cuda(model):
    if next(model.parameters()).is_cuda:
        print("Model on GPU (CUDA)")
    else:
        print("Model on CPU")


def grads_exploitation(input, target, model, criterion, csv_path, bfa_position=0):
    model.eval()
    
    path = os.path.join(csv_path, str(bfa_position))
    os.makedirs(path, exist_ok=True)

    end = time.time()

    # compute output
    output = model(input)
    loss = criterion(output, target)
    
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
        
    loss.backward()
    
    gradients_list = []
    BoxName = []
    layer_idx = 1
    
    for module_idx, (name, module) in enumerate(model.named_modules()):
      if isinstance(module, (quan_Conv2d, quan_Conv1d, quan_Linear)):
        print(name)
        grad_abs = module.weight.grad.data.detach().abs().cpu().numpy()
        grad_flat = grad_abs.flatten().tolist()
        gradients_list.append(grad_flat)
        BoxName.append(f'Layer {layer_idx}')
        layer_idx += 1
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(gradients_list)

    pylab.xticks(range(1, len(BoxName) + 1), BoxName)
    ax.set_xlabel('Layers')
    ax.set_ylabel('Gradient values')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Sauvegarde du graphique
    plt.savefig(os.path.join(path, 'grad_moustache.png'), bbox_inches="tight")

    return 
    

import os
import matplotlib.pyplot as plt
import numpy as np
import pylab

def weights_exploitation(model, csv_path, bfa_position=0, info="8_bits"):
    model.eval()
    
    # Chemin de sauvegarde des resultats
    path = os.path.join(csv_path, str(bfa_position))
    os.makedirs(path, exist_ok=True)

    weights_list = []  # Liste pour stocker les poids de chaque couche
    BoxName = []       # Liste pour stocker les noms des couches
    layer_idx = 1      # Index pour les couches
    
    # Iteration sur les modules du modele
    for module_idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, (quan_Conv2d, quan_Conv1d, quan_Linear)):
            print(name)
            grad_abs = module.weight.data.detach().cpu().numpy()
            weights_list.append(grad_abs.flatten())
            BoxName.append(f'Layer {layer_idx}')
            layer_idx += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Plot des donnees avec un scatter plot
    for i, weights in enumerate(weights_list):
        y_values = [i + 1] * len(weights)  # Definition des y pour chaque layer
        plt.scatter(y_values, weights, alpha=0.6)

    # Configuration des labels pour les axes
    pylab.xticks(range(1, len(BoxName) + 1), BoxName, rotation=45, ha='right')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Weights values')

    # Sauvegarde du graphique de dispersion
    plt.savefig(os.path.join(path, 'weights_scatter'+info+'.png'), bbox_inches="tight")
    plt.close(fig)

    return


if __name__ == '__main__':
    main()
