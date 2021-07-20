'''Train CIFAR10 with PyTorch.'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.utils.data

from models.codinet import *
from utils.dataloader import get_data
from utils.argument import get_args
from utils.utils import *
from utils.metric import MultiLabelAcc
from utils.metric import AverageMetric
from utils.loss import *
from utils.metric import accuracy
from utils.utils import parse_system
from utils.dist_utils import *
from apex.parallel import DistributedDataParallel as DDP


def get_variables(inputs, labels):
    if 'aug' in args.dataset:
        assert len(inputs.shape) == 5
        assert len(labels.shape) == 2
        inputs = inputs.view(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3],
                             inputs.shape[4]).cuda()
        labels = labels.view(-1).cuda()
    else:
        inputs, labels = inputs.cuda(), labels.cuda()
    return inputs, labels


def train_epoch(net, train_loader, logger, epoch):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(dist_tqdm(train_loader)):
        global_step = epoch * len(train_loader) + batch_idx
        inputs, labels = get_variables(inputs, labels)

        result, prob = net(inputs)
        # dist_print(prob.shape)
        # prob: block * batch * 2 * 1 * 1
        loss_CE = criterion_CE(result, labels)
        loss = loss_CE

        if args.loss_lda is not None:
            loss_lda_inter, loss_lda_intra = criterion_ConDiv(prob)
            loss += args.loss_lda * (loss_lda_inter + loss_lda_intra)

        if args.loss_w is not None:
            loss_FL = criterion_FL(prob)
            loss += loss_FL * args.loss_w

        # measure accuracy and record loss
        prec, = accuracy(result, labels, topk=(1,))

        # calc gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.add_scalar('train/prob', mean_reduce(prob[:, :, 0].sum() / prob.shape[0]),
                            global_step=global_step)
        logger.add_scalar('train/prec', mean_reduce(prec), global_step=global_step)
        logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        logger.add_scalar('train/cls', mean_reduce(loss_CE), global_step=global_step)
        logger.add_scalar('train/total', mean_reduce(loss), global_step=global_step)


def valid_epoch(net, valid_loader, logger, epoch):
    counter = MultiLabelAcc()
    probable = AverageMetric()
    path_nums = []

    logits = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dist_tqdm(valid_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            result, prob = net(inputs)

            logits.append(result)

    result = torch.stack(logits, dim=0)
    # print(result.shape)

    all_total = sum_reduce(torch.tensor([counter.total, ], dtype=torch.long).cuda())
    all_correct = sum_reduce(torch.tensor([counter.correct, ], dtype=torch.long).cuda())
    all_prob = mean_reduce(torch.tensor([probable.avg], dtype=torch.float).cuda())
    all_paths = cat_reduce(torch.tensor(path_nums, dtype=torch.long).cuda())

    logger.add_scalar('valid/prec', to_python_float(all_correct) * 100.0 / to_python_float(all_total),
                        global_step=epoch)
    logger.add_scalar('valid/prob', all_prob, global_step=epoch)
    logger.add_scalar('valid/path_num', len(torch.unique(all_paths)), global_step=epoch)

    return to_python_float(all_correct) * 100.0 / to_python_float(all_total)


def save_model(top1, best_acc, epoch, save_path):
    if top1 > best_acc:
        dist_print('Saving best model..')
        state = {'net': net.state_dict(), 'acc': top1, 'epoch': epoch, }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        model_path = os.path.join(save_path, 'ckpt.pth')
        torch.save(state, model_path)
        best_acc = top1
    return best_acc


if __name__ == "__main__":

    args = get_args().parse_args()

    distributed = True if int(os.environ['WORLD_SIZE']) > 1 else False
    
    if distributed:
        assert int(os.environ[
                       'WORLD_SIZE']) == torch.cuda.device_count(), 'It should be the same number of devices and processes'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    best_acc = 0  # best test accuracy
    train_loader, valid_loader, classes = get_data(args.train_bs, args.test_bs, dataset=args.dataset,
                                                  data_root=args.data_root, distributed=distributed,
                                                  aug_repeat=args.aug_repeat)

    cudnn.benchmark = True
    # Model
    dist_print('==> Building model..')

    net = CoDiNet(args.backbone, classes, args.beta, args.finetune, args.freeze_gate, args.hard_sample).cuda()

    dist_print("param size = % MB", count_parameters_in_MB(net))

    if distributed:
        net = net.cuda()
        net = DDP(net, delay_allreduce=True)  # TODO test no delay

    if args.freeze_gate:
        dist_print('==> Load finetune model...')
        net.load_state_dict(load_model(args.resume_path))
    elif args.pretrain:
        net.load_state_dict(load_model(args.model_path))
        dist_print('Load pretrain model')
    else:
        dist_print('No pretrain model')

    logger, save_path = parse_system(args)

    criterion_CE = nn.CrossEntropyLoss().cuda()
    criterion_FL = FLOPSL1Loss(target=args.num_target).cuda()
    criterion_ConDiv = ConDivLoss(args.aug_repeat, args.lda_intra_margin, args.lda_inter_margin).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 85], gamma=0.1)

    for epoch in range(args.epochs):
        dist_print('\nEpoch: %d' % epoch)
    
        scheduler.step(epoch)
    
        train_epoch(net, train_loader, logger, epoch)
    
        top1 = valid_epoch(net, valid_loader, logger, epoch)
    
        synchronize()

        if is_main_process():
            best_acc = save_model(top1, best_acc, epoch, save_path)
        synchronize()
    
        logger.add_scalar('best_acc', best_acc, global_step=epoch)
    
    dist_print('\nRESNET: acc %f' % (best_acc))
    
    torch.set_printoptions(profile="full")
    
    logger.close()
