import pdb
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os, argparse
import cv2
from lib.net_factory_res2net import net_factory
import shutil
import h5py
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # ravd = abs(metric.binary.ravd(pred, gt))
        # hd = metric.binary.hd95(pred, gt)
        # asd = metric.binary.asd(pred, gt)
        return np.array([dice, 0, 0, 0])  # , ravd, hd, asd
    else:
        return np.zeros(4)


def test_all_case(net, base_dir, method="unet", test_list="CVC-300.txt", num_classes=4, patch_size=(384, 384),
                  test_save_path=None, dcf=True):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/test/{}/{}.h5".format(test_list[:-4], item.replace('\n', '')) for item in image_list]

    total_metric = np.zeros((num_classes - 1, 4))

    sized = round(patch_size[0] * 0.5 / 32) * 32

    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in image_list:
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:] / 255.
            label = h5f['label'][:]
            label = np.where((label / 255.0) > 0.5, np.ones_like(label), np.zeros_like(label))
            if "thyroid" in base_dir:
                x, y = image.shape
                image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
                image_d = zoom(image, (sized / patch_size[0], sized / patch_size[1]), order=0)
            else:
                x, y, z = image.shape
                image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=0)
                image_d = zoom(image, (sized / patch_size[0], sized / patch_size[1], 1), order=0)
            input = transforms.ToTensor()(image).unsqueeze(0).float().cuda()
            input_d = transforms.ToTensor()(image_d).unsqueeze(0).float().cuda()

            net.eval()
            with torch.no_grad():
                out = net(input, input_d)
                if dcf:
                    out = out[-1]
                else:
                    out = out[0]
                out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

            # save #
            cv2.imwrite(test_save_path + image_path.split("/")[-1].replace("test_", "").replace(".h5", ".png"),
                        prediction * 255)

            metric = calculate_metric_percase(prediction == 1, label == 1)
            total_metric[0, :] += metric
            f.writelines("{},{},{},{},{}\n".format(
                ids, metric[0], metric[1], metric[2], metric[3]))

        f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
        f.close()
        return total_metric / len(image_list)


def Inference(FLAGS, model, model_path=None):
    time2 = time.time()
    if model_path != None:
        save_mode_path = model_path
    else:
        save_mode_path = FLAGS.exp
    model.load_state_dict(torch.load(save_mode_path)['state'], strict=True)
    time3 = time.time()
    print(time3 - time2)
    print("init weight from {}".format(save_mode_path))

    test_save_path = "{}/Prediction".format(FLAGS.exp.rsplit('/', 1)[0])
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    model.eval()
    for _data_name in data_list:
        print(_data_name)
        save_path = '{}/{}/'.format(test_save_path, _data_name)
        os.makedirs(save_path, exist_ok=True)

        avg_metric = test_all_case(model, base_dir=FLAGS.root_path, method=FLAGS.model,
                                   test_list="{}.txt".format(_data_name), num_classes=num_classes,
                                   patch_size=patch_size, test_save_path=save_path, dcf=FLAGS.dcf)
        print(avg_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        default=None, help='experiment_name')
    parser.add_argument('--root_path', type=str,
                        default='./data/PolypDataset', help='Name of Experiment')
    parser.add_argument('--model', type=str,
                        default="mynet", help='model_name')
    parser.add_argument('--cfa', action='store_false', help='if Cross-level Feature Aggregation')
    parser.add_argument('--dcf', action='store_false', help='if ual-scale Complementary Fusion')
    parser.add_argument('--sc', action='store_false', help='if Scale-enhanced consistency')
    FLAGS = parser.parse_args()

    num_classes = 2
    if "polyp" in FLAGS.root_path:
        data_list = ['CVC-300', 'CVC-ClinicDB', 'Kvasir','CVC-ColonDB','ETIS-LaribPolypDB']
        in_chns = 3
        patch_size = (352, 352)
    elif "ISIC" in FLAGS.root_path:
        data_list = ['test']
        in_chns = 3
        patch_size = (352, 352)
    elif "BrainMRI" in FLAGS.root_path:
        data_list = ['test']
        in_chns = 3
        patch_size = (256, 256)
    model = net_factory(num_classes=num_classes, in_chns=in_chns, SC=FLAGS.sc, CFA=FLAGS.cfa, DCF=FLAGS.dcf).cuda()
    if os.path.isdir(FLAGS.exp):
        model_paths = os.listdir(FLAGS.exp)
        model_paths_fliter = []
        for i in range(len(model_paths)):
            if model_paths[i][:4] == 'iter':
                model_paths_fliter.append(model_paths[i])
        model_paths_fliter = sorted(model_paths_fliter, key=lambda x: int(x.split("_")[1].split('.')[0]))[16:]
        print(model_paths_fliter)
        for i in range(len(model_paths_fliter)):
            Inference(FLAGS, model, FLAGS.exp + model_paths_fliter[i])
    else:
        Inference(FLAGS, model, None)
