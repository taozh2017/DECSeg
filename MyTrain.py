import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.GAN import *
from medpy import metric
from scipy.ndimage import zoom
from utils.dataset import BaseDataSets, TwoStreamBatchSampler, DECAugment
from lib.net_factory_res2net import net_factory
from utils import ramps, losses

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/PolypDataset',
                    help='path to train dataset')
parser.add_argument('--train_save', type=str, default='DEC-Seg',
                    help='the address where the model is saved. if not modified, defaults to debug')
parser.add_argument('--model', type=str, default='mynet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-2, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[352, 352], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--in_chns', type=int, default=3, help='input channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=435,
                    help='labeled data')
# costs
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=6.0, help='consistency_rampup')
# specific
parser.add_argument('--pretrain', type=str, default=None, help='if semi need to set this parameter')
parser.add_argument('--semi', action='store_false', help='if semi')
parser.add_argument('--sc', action='store_false', help='if Scale-enhanced consistency')
parser.add_argument('--spc', action='store_false', help='if Scale-aware Perturbation Consistency')
parser.add_argument('--cc', action='store_false', help='if Cross-generative Consistency')
parser.add_argument('--cfa', action='store_false', help='if Cross-level Feature Aggregation')
parser.add_argument('--dcf', action='store_false', help='if Dual-scale Complementary Fusion')

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "polyp" in dataset:
        ref_dict = {"145": 145, "435": 435}
    elif "ISIC" in dataset:
        ref_dict = {"207": 207, "622": 622}
    elif "BrainMRI" in dataset:
        ref_dict = {"103": 103, "310": 310}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
    # torch.use_deterministic_algorithms(True)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242 超过200个epoch，w就会变成1
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    in_chns = args.in_chns
    size = args.patch_size[0]
    sized = round(size * 0.5 / 32) * 32

    def create_model(in_chns):
        # Network definition
        model = net_factory(num_classes, in_chns, SC=args.sc, CFA=args.cfa, DCF=args.dcf)
        return model

    model = create_model(in_chns).cuda()
    Generator = nn.ModuleList([generator(n_channels=num_classes, n_classes=in_chns, bilinear=False),
                               generator(n_channels=num_classes, n_classes=in_chns, bilinear=False)]).cuda()

    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint["state"], strict=False)
        if 'gen' in checkpoint.keys():
            Generator.load_state_dict(checkpoint['gen'])

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=DECAugment(args.patch_size))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  #
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    if args.cc:
        optimizer = optim.SGD([{'params': model.parameters()},
                               {'params': Generator.parameters()}],
                              lr=base_lr, momentum=0.9, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    MSE_func = nn.MSELoss(reduction='mean')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    consistency_weight = 1.0
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            image_ori, image_down, image_per, image_down_per, label_ori, label_down = (
                sampled_batch["image"],
                sampled_batch["image_down"],
                sampled_batch["image_per"],
                sampled_batch["image_down_per"],
                sampled_batch["label"],
                sampled_batch["label_down"],
            )  # b*3*h*w,b*3*h*w,b*h*w

            image_ori, image_down, image_per, image_down_per, label_ori, label_down = (
                image_ori.cuda(),
                image_down.cuda(),
                image_per.cuda(),
                image_down_per.cuda(),
                label_ori.cuda(),
                label_down.cuda(),
            )

            # outputs for model
            out1, out2, out3 = model(image_ori, image_down)
            out1_soft = torch.softmax(out1, dim=1)
            if args.sc:
                out2_soft = torch.softmax(out2, dim=1)
            if args.dcf:
                out3_soft = torch.softmax(out3, dim=1)

            # sup loss
            sup_loss_list = []
            loss_ce_1 = ce_loss(out1[:args.labeled_bs], label_ori[:][:args.labeled_bs].long())
            loss_dice_1 = dice_loss(out1_soft[:args.labeled_bs], label_ori[:args.labeled_bs].unsqueeze(1))
            sup_loss_list.append(0.5 * (loss_dice_1 + loss_ce_1))

            if args.sc:
                loss_ce_2 = ce_loss(out2[:args.labeled_bs], label_down[:][:args.labeled_bs].long())
                loss_dice_2 = dice_loss(out2_soft[:args.labeled_bs], label_down[:args.labeled_bs].unsqueeze(1))
                sup_loss_list.append(0.5 * (loss_dice_2 + loss_ce_2))

            if args.dcf:
                loss_ce_3 = ce_loss(out3[:args.labeled_bs], label_ori[:][:args.labeled_bs].long())
                loss_dice_3 = dice_loss(out3_soft[:args.labeled_bs], label_ori[:args.labeled_bs].unsqueeze(1))
                sup_loss_list.append(0.5 * (loss_dice_3 + loss_ce_3))

            supervised_loss = torch.sum(torch.stack(sup_loss_list, dim=0))

            # sc loss
            if args.sc:
                scale_loss_list = []
                scale_loss_list.append(
                    F.mse_loss(out1_soft[:args.labeled_bs],
                               F.interpolate(out2_soft[:args.labeled_bs], size=(size, size), mode='bilinear',
                                             align_corners=True), reduction='mean'))
                if not args.semi:
                    sc_loss = torch.sum(torch.stack(scale_loss_list, dim=0))
            else:
                sc_loss = torch.tensor(0.)

            if args.semi:
                if args.spc:
                    out1p, out2p, out3p = model(image_per[args.labeled_bs:], image_down_per[args.labeled_bs:])
                    out1p_soft = torch.softmax(out1p, dim=1)
                    if args.sc:
                        out2p_soft = torch.softmax(out2p, dim=1)
                    if args.dcf:
                        out3p_soft = torch.softmax(out3p, dim=1)

                    consistency_loss_list = []
                    consistency_loss_list.append(MSE_func(out1_soft[args.labeled_bs:], out1p_soft))  # MSE
                    if args.sc:
                        consistency_loss_list.append(MSE_func(out2_soft[args.labeled_bs:], out2p_soft))  # MSE
                    if args.dcf:
                        consistency_loss_list.append(MSE_func(out3_soft[args.labeled_bs:], out3p_soft))  # MSE
                    consistency_loss = torch.sum(torch.stack(consistency_loss_list, dim=0))
                else:
                    consistency_loss = torch.tensor(0.)

                if args.sc:
                    scale_loss_list.append(
                        F.mse_loss(out1_soft[args.labeled_bs:],
                                   F.interpolate(out2_soft[args.labeled_bs:], size=(size, size), mode='bilinear',
                                                 align_corners=True), reduction='mean'))
                    if args.spc:
                        scale_loss_list.append(
                            F.mse_loss(out1p_soft, F.interpolate(out2p_soft, size=(size, size), mode='bilinear',
                                                                 align_corners=True), reduction='mean'))
                    sc_loss = torch.sum(torch.stack(scale_loss_list, dim=0))

                if args.spc and args.cc:
                    cc_loss_list = []
                    if args.dcf:
                        real_A = [out3[args.labeled_bs:], out3p]
                    else:
                        real_A = [out1[args.labeled_bs:], out1p]

                    real_B = [image_per[args.labeled_bs:], image_ori[args.labeled_bs:]]

                    fake_B = [Generator[0](real_A[0]), Generator[1](real_A[1])]

                    cc_loss_list.append(MSE_func(fake_B[0], real_B[0]))
                    cc_loss_list.append(MSE_func(fake_B[1], real_B[1]))
                    cc_loss = torch.sum(torch.stack(cc_loss_list, dim=0))
                else:
                    cc_loss = torch.tensor(0.)
            else:
                consistency_loss = torch.tensor(0.)
                cc_loss = torch.tensor(0.)

            # consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + consistency_weight * (consistency_loss + sc_loss + cc_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_sup', supervised_loss, iter_num)
            writer.add_scalar('info/loss_spc', consistency_loss, iter_num)
            writer.add_scalar('info/loss_sc', sc_loss, iter_num)
            writer.add_scalar('info/loss_cc', cc_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_sup: %f, loss_spc: %f, loss_sc: %f, loss_cc: %f' %
                (iter_num, loss.item(), supervised_loss.item(), consistency_weight * consistency_loss.item(),
                 consistency_weight * sc_loss.item(), consistency_weight * cc_loss.item()))

            if iter_num % 20 == 0:
                writer.add_image('train/11Image_ori', image_ori[-1, :, :, :], iter_num)

                labs = label_ori[-1, ...].unsqueeze(0) * 255
                writer.add_image('train/12GroundTruth', labs, iter_num)
                writer.add_image('train/13Prediction_1',
                                 torch.argmax(out1_soft, dim=1, keepdim=True)[-1, ...].detach(), iter_num)
                if args.sc:
                    writer.add_image('train/14Prediction_2',
                                     torch.argmax(out2_soft, dim=1, keepdim=True)[-1, ...].detach(), iter_num)
                if args.dcf:
                    writer.add_image('train/15Prediction_3',
                                     torch.argmax(out3_soft, dim=1, keepdim=True)[-1, ...].detach(), iter_num)

                if args.semi and args.spc:
                    writer.add_image('train/21Image_per', image_per[-1, :, :, :], iter_num)
                    writer.add_image('train/22Prediction_per1',
                                     torch.argmax(out1p_soft, dim=1, keepdim=True)[-1, ...].detach(), iter_num)
                    if args.sc:
                        writer.add_image('train/23Prediction_per2',
                                         torch.argmax(out2p_soft, dim=1, keepdim=True)[-1, ...].detach(), iter_num)
                    if args.dcf:
                        writer.add_image('train/24Prediction_per3',
                                         torch.argmax(out3p_soft, dim=1, keepdim=True)[-1, ...].detach(), iter_num)
                    if args.cc:
                        writer.add_image('train/31Gen1',
                                         fake_B[0][-1, ...].detach(), iter_num)
                        writer.add_image('train/31Gen2',
                                         fake_B[1][-1, ...].detach(), iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_slice(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,
                        patch_size=args.patch_size, dcf=args.dcf)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance.item()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(performance.item(), 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    state = {'state': model.state_dict()}
                    if args.cc:
                        state['gen'] = Generator.state_dict()
                    torch.save(state, save_mode_path)
                    torch.save(state, save_best)
                elif best_performance - performance < 0.01:
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(performance.item(), 4)))
                    state = {'state': model.state_dict()}
                    if args.cc:
                        state['gen'] = Generator.state_dict()
                    torch.save(state, save_mode_path)

                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                state = {'state': model.state_dict()}
                if args.cc:
                    state['gen'] = Generator.state_dict()
                torch.save(state, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        return dice, 0
    else:
        return 0, 0


def test_single_slice(image, label, net, classes, patch_size=[384, 384], dcf=True):
    # pdb.set_trace()
    image = image.permute(0, 3, 1, 2)
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    slice = image
    z, x, y = slice.shape
    sized = round(patch_size[0] * 0.5 / 32) * 32
    slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
    image_d = zoom(slice, (1, sized / patch_size[0], sized / patch_size[1]), order=0)
    input_d = torch.from_numpy(image_d).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if dcf:
            out = net(input, input_d)[-1]
        else:
            out = net(input, input_d)[0]

        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def val(model, val_loader, dcf):
    model.eval()
    dice_avg = 0.0
    for i_batch, sampled_batch in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, image_down, label = sampled_batch["image"], sampled_batch["image_down"], sampled_batch["label"]
        image = image.cuda()  # b*c*h*w
        image_down = image_down.cuda()  # b*c*h*w
        label = label.cuda()  # b*h*w

        # ---- forward ----
        with torch.no_grad():
            out1, out2, out3 = model(image, image_down)

        if dcf:
            res = out3
        else:
            res = out1
        # ---- loss function ----
        res = torch.argmax(torch.softmax(res, dim=1), dim=1).squeeze(0)  # H*W
        dice = losses.Dice(res, label[0])

        # ---- recording loss ----
        dice_avg += dice
    return dice_avg / len(val_loader)


if __name__ == "__main__":
    # set seed
    same_seeds(args.seed)

    # -- build save dir -- #
    snapshot_path = os.path.join("snatplot",
                                 datetime.now().strftime("%m%d") + "_" + args.train_save)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
        os.makedirs(snapshot_path + '/code')
    shutil.copy(os.path.abspath(__file__), os.path.join(snapshot_path, 'train.py'))
    shutil.copytree(os.path.join(os.getcwd(), 'lib'), snapshot_path + '/code/lib')
    shutil.copytree(os.path.join(os.getcwd(), 'utils'), snapshot_path + '/code/utils')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)

    print("train finish")
