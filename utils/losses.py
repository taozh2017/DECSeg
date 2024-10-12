import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import dequeue_and_enqueue_prior
import cv2


def Dice(pred, mask):
    inter = (pred * mask).sum()
    union = (pred + mask).sum()
    dice = (2 * inter + 1e-10) / (union + 1e-10)
    return dice


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def bceiouloss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='mean')

    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def mse_loss(inputs, targets, no_sigmoid=False):
    if not no_sigmoid:
        inputs = torch.sigmoid(inputs)
        targets = torch.sigmoid(targets)

    wmse = F.mse_loss(inputs, targets, reduction='mean')
    return wmse


def pixel_contra_labeled(
        rep,  # [8, 256, 24, 24]
        label,  # bs*h*w
        pred,  # bs*c*h*w
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter=0
):
    '''
    对于labeled而言，对于前景，先按照gt筛选，cnn预测高置信度正确的加入mb，预测不正确的视为困难样本作为锚。
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    temp = 0.5
    alpha_t = 80
    num_queries = 50
    num_negatives = 256

    rep = rep.permute(0, 2, 3, 1)  # bs*24*24*256
    b, h, w, c = rep.shape

    if label.shape[1] != h or label.shape[2] != w:
        label = F.interpolate(label, size=(h, w), mode='nearest')  # bhw
    if pred.shape[1] != h or pred.shape[2] != w:
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)  # b1hw

    pred = torch.cat((1 - pred, pred), dim=1)
    fea_list = []
    for i in range(2):
        mask1 = (label == i).squeeze(1)
        pred_c = pred[:, i]
        if mask1.sum() > 0:
            thresh1 = max(np.percentile(pred_c[mask1].cpu().detach().numpy().flatten(), 20), 0.5)
            mask2 = (pred_c >= thresh1)
        else:
            mask2 = torch.ones_like(mask1)

        if torch.sum(prototype[i]) != 0 and (mask1 * mask2).sum() > 0:
            # pdb.set_trace()
            feat_dis = torch.cosine_similarity(rep[mask1 * mask2], prototype[i])  # 一维
            thresh2 = np.percentile(
                feat_dis.cpu().detach().numpy().flatten(), 20
            )  # 越接近1，相似度越高
            mask3 = (feat_dis > thresh2)  # 一维 取了更好的20%
        else:
            mask3 = torch.ones((mask1 * mask2).sum()).bool()
        fea = rep[mask1 * mask2][mask3]
        fea_list.append(fea)
        keys = (pred_c[mask1 * mask2][mask3])
        dequeue_and_enqueue_prior(feats=fea.detach(),
                                  keys=keys.detach(),  # keys 可以改一下看看
                                  queue=memobank[i],
                                  queue_key=keys_mb[i],
                                  queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i],
                                  queue_size=queue_size[i])

    if len(fea_list[0]) > 0 and len(fea_list[1] > 0):
        reco_loss = []
        for i in range(2):
            query_id = torch.randint(len(fea_list[i]), size=(num_queries,))
            query = fea_list[i][query_id]
            key = memobank[1 - i][0]
            key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
            key = key[key_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
        return (reco_loss[0] + reco_loss[1]) / 2
    else:
        return torch.tensor(0.)

    # if prototype is not None:
    #     if not (prototype == 0).all():
    #         ema_decay = min(1 - 1 / i_iter, 0.999)
    #         positive_feat = (1 - ema_decay) * positive_feat + ema_decay * prototype
    #     prototype == positive_feat.clone()


def pixel_contra_labeled_wsam(
        rep,  # [8, 256, 24, 24]
        label,  # bs*h*w
        pred,  # bs*c*h*w
        cnn,
        sam,
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter=0
):
    '''
    对于labeled而言，对于前景，先按照gt筛选，cnn预测高置信度正确的加入mb，预测不正确的视为困难样本作为锚。
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    temp = 0.5
    alpha_t = 80
    num_queries = 50
    num_negatives = 256

    rep = rep.permute(0, 2, 3, 1)  # bs*24*24*256
    b, h, w, c = rep.shape

    if label.shape[1] != h or label.shape[2] != w:
        label = F.interpolate(label, size=(h, w), mode='nearest')  # bhw
    if pred.shape[1] != h or pred.shape[2] != w:
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)  # b1hw
    if sam.shape[1] != h or sam.shape[2] != w:
        sam = F.interpolate(sam, size=(h, w), mode='nearest')  # b1hw
    if cnn.shape[1] != h or cnn.shape[2] != w:
        cnn = F.interpolate(torch.round(cnn), size=(h, w), mode='nearest')  # b1hw

    pred = torch.cat((1 - pred, pred), dim=1)
    sam = torch.cat((1 - sam, sam), dim=1)
    cnn = torch.cat((1 - cnn, cnn), dim=1)
    fea_tough_list = []
    fea_easy_list = []
    for i in range(2):
        mask1 = (label == i).squeeze(1)
        pred_c = pred[:, i]
        mask4 = (sam[:, i] * cnn[:, i]).bool()
        fea_tough_list.append(rep[mask1 * (~mask4)])
        fea_easy_list.append(rep[mask1 * mask4])
        # pdb.set_trace()
        dequeue_and_enqueue_prior(feats=rep[mask1 * mask4].detach(),
                                  keys=pred_c[mask1 * mask4].detach(),  # keys 可以改一下看看
                                  queue=memobank[i],
                                  queue_key=keys_mb[i],
                                  queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i],
                                  queue_size=queue_size[i])

    reco_loss = []
    for i in range(2):
        if len(fea_tough_list[i]) != 0 and len(fea_easy_list[i] != 0):
            query_id_tough = torch.randint(len(fea_tough_list[i]), size=(int(num_queries / 2),))
            query_tough = fea_tough_list[i][query_id_tough]
            query_id_easy = torch.randint(len(fea_easy_list[i]), size=(int(num_queries / 2),))
            query_easy = fea_easy_list[i][query_id_easy]
            query = torch.cat((query_easy, query_tough), dim=0)
        elif len(fea_tough_list[i]) == 0:
            query_id_easy = torch.randint(len(fea_easy_list[i]), size=(num_queries,))
            query = fea_easy_list[i][query_id_easy]
        elif len(fea_easy_list[i]) == 0:
            query_id_tough = torch.randint(len(fea_tough_list[i]), size=(num_queries,))
            query = fea_tough_list[i][query_id_tough]
        else:
            continue
        key = memobank[1 - i][0]
        if len(key) == 0:
            continue
        key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
        key = key[key_id].reshape(num_queries, num_negatives, c)
        all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
        seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
        reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()).unsqueeze(0))
        # pdb.set_trace()
    if len(reco_loss) > 0:
        # pdb.set_trace()
        return torch.mean(torch.cat(reco_loss, dim=0))
    else:
        return torch.tensor(0.)


def pixelcontra_lwsam_soft(
        rep,  # bs*c*h*w
        label,  # bs*h*w
        pred,  # bs*2*h*w
        cnn,  # bs*2*h*w
        sam,  # bs*1*h*w
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter=0
):
    '''
    对于labeled而言，对于前景，先按照gt筛选，cnn预测高置信度正确的加入mb，预测不正确的视为困难样本作为锚。
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    temp = 0.5
    alpha_t = 80
    num_queries = 50
    num_negatives = 256

    rep = rep.permute(0, 2, 3, 1)  # bs*24*24*256
    b, h, w, c = rep.shape

    if label.shape[1] != h or label.shape[2] != w:
        label = F.interpolate(label.unsqueeze(1), size=(h, w), mode='nearest')  # b1hw
    if pred.shape[1] != h or pred.shape[2] != w:
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)  # b1hw
    if sam.shape[1] != h or sam.shape[2] != w:
        sam = F.interpolate(sam, size=(h, w), mode='nearest')  # b1hw
    if cnn.shape[1] != h or cnn.shape[2] != w:
        cnn = F.interpolate(torch.argmax(cnn, dim=1, keepdim=True).float(), size=(h, w), mode='nearest')  # b2hw

    sam = torch.cat((1 - sam, sam), dim=1)
    cnn = torch.cat((1 - cnn, cnn), dim=1)
    fea_tough_list = []
    fea_easy_list = []
    for i in range(2):
        mask1 = (label == i).squeeze(1)
        pred_c = pred[:, i]
        mask4 = (sam[:, i] * cnn[:, i]).bool()
        fea_tough_list.append(rep[mask1 * (~mask4)])
        fea_easy_list.append(rep[mask1 * mask4])
        # pdb.set_trace()
        dequeue_and_enqueue_prior(feats=rep[mask1 * mask4].detach(),
                                  keys=pred_c[mask1 * mask4].detach(),  # keys 可以改一下看看
                                  queue=memobank[i],
                                  queue_key=keys_mb[i],
                                  queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i],
                                  queue_size=queue_size[i])

    reco_loss = []
    for i in range(2):
        if len(fea_easy_list[i]) != 0:
            query_id_easy = torch.randint(len(fea_easy_list[i]), size=(num_queries,))
            query = fea_easy_list[i][query_id_easy]
        else:
            continue
        key = memobank[1 - i][0]
        if len(key) == 0:
            continue
        key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
        key = key[key_id].reshape(num_queries, num_negatives, c)
        all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
        seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
        reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()).unsqueeze(0))
        # pdb.set_trace()
    if len(reco_loss) > 0:
        # pdb.set_trace()
        return torch.mean(torch.cat(reco_loss, dim=0))
    else:
        return torch.tensor(0.)


def pixel_contra_unlabeled(
        rep,  # [8, 256, 24, 24]
        pred,
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter
):
    '''
    对于unlabeled而言，要找更像前景的点做过滤，fus和fuw一致性更高，且和cnn预测一致的越像前景
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 20
    num_queries = 50
    num_negatives = 256

    rep = rep.permute(0, 2, 3, 1)  # bs*12*12*128
    b, h, w, c = rep.shape
    if pred.shape[1] != h or pred.shape[2] != w:
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

    # TODO positive filter
    high_thresh1 = np.percentile(
        pred.cpu().detach().numpy().flatten(), 100 - alpha_t
    )
    mask_pos1 = (pred > thre).squeeze(1)
    mask_pos2 = (pred > high_thresh1).squeeze(1)

    # if torch.sum(prototype[0][0]) != 0 and torch.sum(prototype[0][1]) != 0:
    #     pos_feat_dis1 = torch.cosine_similarity(rep[mask_pos1], prototype[0][0])  # 一维
    #     pos_feat_thresh1 = np.percentile(
    #         pos_feat_dis1.cpu().detach().numpy().flatten(), 50
    #     )  # 越接近1，相似度越高
    #     pos_mask_feat1 = (pos_feat_dis1 > pos_feat_thresh1)  # 一维 取了更好的20%
    # else:
    #     pos_mask_feat1 = torch.ones(mask_pos1.sum()).bool()

    # TODO negtive filter
    low_thresh1 = np.percentile(
        pred.cpu().detach().numpy().flatten(), alpha_t
    )
    mask_neg1 = (pred < thre).squeeze(1)
    mask_neg2 = (pred < low_thresh1).squeeze(1)
    # if torch.sum(prototype[1][0]) != 0 and torch.sum(prototype[1][1]) != 0:
    #     neg_feat_dis1 = torch.cosine_similarity(rep[mask_neg1], prototype[1][0])  # 一维
    #     neg_feat_thresh1 = np.percentile(
    #         neg_feat_dis1.cpu().detach().numpy().flatten(), 50
    #     )  # 越接近1，相似度越高
    #     neg_mask_feat1 = (neg_feat_dis1 > neg_feat_thresh1)  # 一维 取了更好的20%
    # else:
    #     neg_mask_feat1 = torch.ones(mask_neg1.sum()).bool()

    neg_pixels1 = rep[mask_neg1 * mask_neg2]  # [neg_mask_feat1]
    neg_keys = (1 - pred.squeeze(dim=1)[mask_neg1 * mask_neg2]).detach()
    dequeue_and_enqueue_prior(feats=neg_pixels1.detach(),
                              keys=neg_keys,
                              queue=memobank[0],
                              queue_key=keys_mb[0],
                              queue_ptr=queue_prtlis[0],
                              queue_size=queue_size[0])

    if (mask_pos1 * mask_pos2).sum() > 0 and len(neg_pixels1) > 0:
        anchor = rep[mask_pos1 * mask_pos2]  # TODO 用容易样本表示类特征
        positive_feat = torch.mean(anchor, dim=0, keepdim=True)
        if prototype is not None:
            if not (prototype == 0).all():
                ema_decay = min(1 - 1 / i_iter, 0.999)
                prototype = (1 - ema_decay) * positive_feat + ema_decay * prototype
            else:
                prototype == positive_feat.clone()

        pos_id = torch.randint(len(anchor), size=(num_queries,))
        anchor_feat = anchor[pos_id]
        with torch.no_grad():
            positive_feat = prototype.unsqueeze(0).repeat(num_queries, 1, 1)
            negative_feat = memobank[0][0]
            neg_id = torch.randint(len(negative_feat), size=(num_queries * num_negatives,))
            negative_feat = negative_feat[neg_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((positive_feat, negative_feat), dim=1)
        seg_logits = torch.cosine_similarity(
            anchor_feat.unsqueeze(1), all_feat, dim=2
        )
        reco_loss = F.cross_entropy(
            seg_logits / temp, torch.zeros(num_queries).long().cuda()
        )
    else:
        reco_loss = torch.tensor(0.)
    return reco_loss, prototype


def pixel_contra_unlabeled_dv(
        rep_w,  # [8, 256, 24, 24]
        pred_w,
        rep_s,  # [8, 256, 24, 24]
        pred_s,
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter
):
    '''
    对于unlabeled而言，要找更像前景的点做过滤，fus和fuw一致性更高，且和cnn预测一致的越像前景
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 50
    num_queries = 50
    num_negatives = 256

    rep_w = rep_w.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_s = rep_s.permute(0, 2, 3, 1)  # bs*12*12*128
    b, h, w, c = rep_w.shape
    if pred_w.shape[1] != h or pred_w.shape[2] != w:
        pred_w = F.interpolate(pred_w, size=(h, w), mode='bilinear', align_corners=True)
    if pred_s.shape[1] != h or pred_s.shape[2] != w:
        pred_s = F.interpolate(pred_s, size=(h, w), mode='bilinear', align_corners=True)

    pred_w = torch.cat((1 - pred_w, pred_w), dim=1)
    pred_s = torch.cat((1 - pred_s, pred_s), dim=1)
    fea_list = []
    for i in range(2):
        pred_w_c = pred_w[:, i]
        pred_s_c = pred_s[:, i]
        mask1 = ((pred_w_c > thre) * (pred_s_c > thre))
        if mask1.sum() > 0:
            thresh1 = np.percentile(
                torch.cat((pred_w_c[mask1], pred_s_c[mask1]), dim=0).cpu().detach().numpy().flatten(), 20)
            mask2 = ((pred_w_c >= thresh1) * (pred_s_c >= thresh1))
        else:
            mask2 = torch.ones_like(mask1)
        if torch.sum(prototype[i]) != 0 and (mask1 * mask2).sum() > 0:
            feat_dis_w = torch.cosine_similarity(rep_w[mask1 * mask2], prototype[i])
            feat_dis_s = torch.cosine_similarity(rep_s[mask1 * mask2], prototype[i])
            thresh2 = np.percentile(
                torch.cat((feat_dis_w, feat_dis_s)).cpu().detach().numpy().flatten(), 20
            )  # 越接近1，相似度越高
            mask3 = (feat_dis_w > thresh2) * (feat_dis_s > thresh2)  # 一维 取了更好的20%
        else:
            mask3 = torch.ones((mask1 * mask2).sum()).bool()
        fea = torch.cat((rep_w[mask1 * mask2][mask3], rep_s[mask1 * mask2][mask3]), dim=0)
        fea_list.append(fea)
        keys = (((pred_w_c + pred_s_c) / 2)[mask1 * mask2][mask3]).repeat(2)
        dequeue_and_enqueue_prior(feats=fea.detach(), keys=keys.detach(),
                                  queue=memobank[i], queue_key=keys_mb[i], queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i], queue_size=queue_size[i])

    if len(fea_list[0]) > 0 and len(fea_list[1] > 0):
        reco_loss = []
        for i in range(2):
            query_id = torch.randint(len(fea_list[i]), size=(num_queries,))
            query = fea_list[i][query_id]
            key = memobank[1 - i][0]
            key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
            key = key[key_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
            # pdb.set_trace()
        return (reco_loss[0] + reco_loss[1]) / 2
    else:
        return torch.tensor(0.)


def pcul_dv_soft(
        rep_w,  # [8, 256, 24, 24]
        label_w,
        pred_w,
        rep_s,  # [8, 256, 24, 24]
        label_s,
        pred_s,
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
):
    '''
    对于unlabeled而言，要找更像前景的点做过滤，fus和fuw一致性更高，且和cnn预测一致的越像前景
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 50
    num_queries = 50
    num_negatives = 256

    rep_w = rep_w.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_s = rep_s.permute(0, 2, 3, 1)  # bs*12*12*128
    b, h, w, c = rep_w.shape
    if label_w.shape[1] != h or label_w.shape[2] != w:
        label_w = F.interpolate(label_w.unsqueeze(1).float(), size=(h, w), mode='nearest')  # bhw
    if label_s.shape[1] != h or label_s.shape[2] != w:
        label_s = F.interpolate(label_s.unsqueeze(1).float(), size=(h, w), mode='nearest')  # bhw
    if pred_w.shape[1] != h or pred_w.shape[2] != w:
        pred_w = F.interpolate(pred_w, size=(h, w), mode='bilinear', align_corners=True)
    if pred_s.shape[1] != h or pred_s.shape[2] != w:
        pred_s = F.interpolate(pred_s, size=(h, w), mode='bilinear', align_corners=True)

    fea_list = []
    for i in range(2):
        mask1 = ((label_w == i) * (label_s == i)).squeeze(1)
        pred_w_c = pred_w[:, i]
        pred_s_c = pred_s[:, i]
        if mask1.sum() > 0:
            thresh1 = np.percentile(
                torch.cat((pred_w_c[mask1], pred_s_c[mask1]), dim=0).cpu().detach().numpy().flatten(), 50)
            mask2 = ((pred_w_c >= thresh1) * (pred_s_c >= thresh1))
        else:
            mask2 = torch.ones_like(mask1)
        if torch.sum(prototype[i]) != 0 and (mask1 * mask2).sum() > 0:
            feat_dis_w = torch.cosine_similarity(rep_w[mask1 * mask2], prototype[i])
            feat_dis_s = torch.cosine_similarity(rep_s[mask1 * mask2], prototype[i])
            thresh2 = np.percentile(
                torch.cat((feat_dis_w, feat_dis_s)).cpu().detach().numpy().flatten(), 50
            )  # 越接近1，相似度越高
            mask3 = (feat_dis_w > thresh2) * (feat_dis_s > thresh2)  # 一维 取了更好的20%
        else:
            mask3 = torch.ones((mask1 * mask2).sum()).bool()
        fea = torch.cat((rep_w[mask1 * mask2][mask3], rep_s[mask1 * mask2][mask3]), dim=0)
        fea_list.append(fea)
        keys = (((pred_w_c + pred_s_c) / 2)[mask1 * mask2][mask3]).repeat(2)
        dequeue_and_enqueue_prior(feats=fea.detach(), keys=keys.detach(),
                                  queue=memobank[i], queue_key=keys_mb[i], queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i], queue_size=queue_size[i])

    if len(fea_list[0]) > 0 and len(fea_list[1] > 0):
        reco_loss = []
        for i in range(2):
            query_id = torch.randint(len(fea_list[i]), size=(num_queries,))
            query = fea_list[i][query_id]
            key = memobank[1 - i][0]
            key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
            key = key[key_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
            # pdb.set_trace()
        return (reco_loss[0] + reco_loss[1]) / 2
    else:
        return torch.tensor(0.)


def pcon_ul_dv_wsam(
        rep_w,  # [8, 256, 24, 24]
        pred_w,
        cnn_w,
        sam_w,
        rep_s,  # [8, 256, 24, 24]
        pred_s,
        cnn_s,
        sam_s,
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter
):
    '''
    对于unlabeled而言，要找更像前景的点做过滤，fus和fuw一致性更高，且和cnn预测一致的越像前景
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 50
    num_queries = 50
    num_negatives = 256

    rep_w = rep_w.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_s = rep_s.permute(0, 2, 3, 1)  # bs*12*12*128
    b, h, w, c = rep_w.shape
    if pred_w.shape[1] != h or pred_w.shape[2] != w:
        pred_w = F.interpolate(pred_w, size=(h, w), mode='bilinear', align_corners=True)
    if pred_s.shape[1] != h or pred_s.shape[2] != w:
        pred_s = F.interpolate(pred_s, size=(h, w), mode='bilinear', align_corners=True)
    if cnn_w.shape[1] != h or cnn_w.shape[2] != w:
        cnn_w = F.interpolate(torch.round(cnn_w), size=(h, w), mode='bilinear', align_corners=True)
    if cnn_s.shape[1] != h or cnn_s.shape[2] != w:
        cnn_s = F.interpolate(torch.round(cnn_s), size=(h, w), mode='bilinear', align_corners=True)
    if sam_w.shape[1] != h or sam_w.shape[2] != w:
        sam_w = F.interpolate(sam_w, size=(h, w), mode='bilinear', align_corners=True)
    if sam_s.shape[1] != h or sam_s.shape[2] != w:
        sam_s = F.interpolate(sam_s, size=(h, w), mode='bilinear', align_corners=True)

    pred_w = torch.cat((1 - pred_w, pred_w), dim=1)
    pred_s = torch.cat((1 - pred_s, pred_s), dim=1)
    cnn_w = torch.cat((1 - cnn_w, cnn_w), dim=1)
    cnn_s = torch.cat((1 - cnn_s, cnn_s), dim=1)
    sam_w = torch.cat((1 - sam_w, sam_w), dim=1)
    sam_s = torch.cat((1 - sam_s, sam_s), dim=1)
    fea_list = []
    for i in range(2):
        pred_w_c = pred_w[:, i]
        pred_s_c = pred_s[:, i]
        mask1 = ((pred_w_c > thre) * (pred_s_c > thre))
        mask2 = (sam_w[:, i] * cnn_w[:, i] * sam_s[:, i] * cnn_s[:, i]).bool()
        fea = torch.cat((rep_w[mask1 * mask2], rep_s[mask1 * mask2]), dim=0)
        fea_list.append(fea)
        keys = (((pred_w_c + pred_s_c) / 2)[mask1 * mask2]).repeat(2)
        dequeue_and_enqueue_prior(feats=fea.detach(), keys=keys.detach(),
                                  queue=memobank[i], queue_key=keys_mb[i], queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i], queue_size=queue_size[i])

    reco_loss = []
    for i in range(2):
        if len(fea_list[i]) > 0:
            query_id = torch.randint(len(fea_list[i]), size=(num_queries,))
            query = fea_list[i][query_id]
        else:
            continue
        key = memobank[1 - i][0]
        if len(key) == 0:
            continue
        key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
        key = key[key_id].reshape(num_queries, num_negatives, c)
        all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
        seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
        reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()).unsqueeze(0))
        # pdb.set_trace()
    if len(reco_loss) > 0:
        return torch.mean(torch.cat(reco_loss, dim=0))
    else:
        return torch.tensor(0.)


def pcon_ul_dv_wsam_soft(
        rep_w,  # [8, 256, 24, 24]
        pred_w,
        cnn_w,
        sam_w,
        rep_s,  # [8, 256, 24, 24]
        pred_s,
        cnn_s,
        sam_s,
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter,
        rep_sam_w=None,
        rep_sam_s=None
):
    '''
    对于unlabeled而言，要找更像前景的点做过滤，fus和fuw一致性更高，且和cnn预测一致的越像前景
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 50
    num_queries = 50
    num_negatives = 256

    rep_w = rep_w.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_s = rep_s.permute(0, 2, 3, 1)  # bs*12*12*128
    if rep_sam_w != None and rep_sam_s != None:
        rep_sam_w = rep_sam_w.permute(0, 2, 3, 1)  # bs*12*12*128
        rep_sam_s = rep_sam_s.permute(0, 2, 3, 1)  # bs*12*12*128

    b, h, w, c = rep_w.shape
    if pred_w.shape[1] != h or pred_w.shape[2] != w:
        pred_w = F.interpolate(pred_w, size=(h, w), mode='bilinear', align_corners=True)
    if pred_s.shape[1] != h or pred_s.shape[2] != w:
        pred_s = F.interpolate(pred_s, size=(h, w), mode='bilinear', align_corners=True)
    if cnn_w.shape[1] != h or cnn_w.shape[2] != w:
        cnn_w = F.interpolate(torch.argmax(cnn_w, dim=1, keepdim=True).float(), size=(h, w), mode='bilinear',
                              align_corners=True)
    if cnn_s.shape[1] != h or cnn_s.shape[2] != w:
        cnn_s = F.interpolate(torch.argmax(cnn_s, dim=1, keepdim=True).float(), size=(h, w), mode='bilinear',
                              align_corners=True)
    if sam_w.shape[1] != h or sam_w.shape[2] != w:
        sam_w = F.interpolate(sam_w, size=(h, w), mode='bilinear', align_corners=True)
    if sam_s.shape[1] != h or sam_s.shape[2] != w:
        sam_s = F.interpolate(sam_s, size=(h, w), mode='bilinear', align_corners=True)

    cnn_w = torch.cat((1 - cnn_w, cnn_w), dim=1)
    cnn_s = torch.cat((1 - cnn_s, cnn_s), dim=1)
    sam_w = torch.cat((1 - sam_w, sam_w), dim=1)
    sam_s = torch.cat((1 - sam_s, sam_s), dim=1)
    fea_list = []
    for i in range(2):
        pred_w_c = pred_w[:, i]
        pred_s_c = pred_s[:, i]
        mask1 = ((pred_w_c > thre) * (pred_s_c > thre))
        mask2 = (sam_w[:, i] * cnn_w[:, i] * sam_s[:, i] * cnn_s[:, i]).bool()
        fea = torch.cat((rep_w[mask1 * mask2], rep_s[mask1 * mask2]), dim=0)
        fea_list.append(fea)
        keys = (((pred_w_c + pred_s_c) / 2)[mask1 * mask2]).repeat(2)
        dequeue_and_enqueue_prior(feats=fea.detach(), keys=keys.detach(),
                                  queue=memobank[i], queue_key=keys_mb[i], queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i], queue_size=queue_size[i])
        if rep_sam_w != None and rep_sam_s != None:
            fea_sam = torch.cat((rep_sam_w[mask1 * mask2], rep_sam_s[mask1 * mask2]), dim=0)
            dequeue_and_enqueue_prior(feats=fea_sam.detach(),
                                      keys=keys.detach(),  # keys 可以改一下看看
                                      queue=memobank[i + 2],
                                      queue_key=keys_mb[i + 2],
                                      queue_time=queue_time[i + 2],
                                      queue_ptr=queue_prtlis[i + 2],
                                      queue_size=queue_size[i + 2])

    reco_loss = []
    for i in range(2):
        if len(fea_list[i]) > 0:
            query_id = torch.randint(len(fea_list[i]), size=(num_queries,))
            query = fea_list[i][query_id]
        else:
            continue
        key = memobank[1 - i][0]
        if len(key) == 0:
            continue
        key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
        key = key[key_id].reshape(num_queries, num_negatives, c)
        all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
        seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
        reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()).unsqueeze(0))
        # pdb.set_trace()
    if len(reco_loss) > 0:
        return torch.mean(torch.cat(reco_loss, dim=0))
    else:
        return torch.tensor(0.)


def paircontra(rep_cnn, rep_sam, label, pred):
    thre = 0.5
    temp = 0.5
    num_queries = 50
    num_negatives = 256

    rep_cnn = rep_cnn.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_sam = rep_sam.permute(0, 2, 3, 1)  # bs*12*12*128
    b, h, w, c = rep_cnn.shape

    if label.shape[1] != h or label.shape[2] != w:
        label = F.interpolate(label, size=(h, w), mode='nearest').squeeze(1)  # bhw
    if pred.shape[1] != h or pred.shape[2] != w:
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

    pred = torch.cat((1 - pred, pred), dim=1)
    mask_list = []
    for i in range(2):
        pred_c = pred[:, i]
        mask1 = (label == i)
        if mask1.sum() > 0:
            thresh1 = max(np.percentile(pred_c[mask1].cpu().detach().numpy().flatten(), 20), thre)
            mask2 = (pred_c >= thresh1)
        else:
            mask2 = torch.ones_like(mask1)
        mask_list.append(mask1 * mask2)
        # mask_list.append(mask1)

    reco_loss = []
    if mask_list[0].sum() > 0 and mask_list[1].sum() > 0:
        for i in range(2):
            query = rep_cnn[mask_list[i]]
            pos = rep_sam[mask_list[i]]
            query_id = torch.randint(len(query), size=(num_queries,))
            query = query[query_id]
            pos = pos[query_id]  # 50*256
            neg = rep_sam[mask_list[1 - i]]
            neg_id = torch.randint(len(neg), size=(num_queries * num_negatives,))
            neg = neg[neg_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((pos.unsqueeze(1), neg), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()).unsqueeze(0))
            # pdb.set_trace()
        return torch.mean(torch.cat(reco_loss, dim=0))
    else:
        return torch.tensor(0.)


def pc(
        rep,  # [8, 256, 24, 24]
        label,  # bs*h*w
        pred,  # bs*c*h*w
        memobank,  # 2*[]*0*128
        keys_mb,  # 4*[]*0*1
        queue_time,
        queue_prtlis,  # 2*1（0）
        queue_size,  # 2*1(30000)
        prototype,  # 2*2*1*128
        i_iter=0
):
    '''
    对于labeled而言，对于前景，先按照gt筛选，cnn预测高置信度正确的加入mb，预测不正确的视为困难样本作为锚。
    只有前景有类特征，背景直接从mb中随机选，那mb中就必须有多样化的特征，特征筛选上最好就是在可入选的范围内替换掉分数最高的 TODO
    '''
    temp = 0.5
    alpha_t = 80
    num_queries = 50
    num_negatives = 256

    rep = rep.permute(0, 2, 3, 1)  # bs*24*24*256
    b, h, w, c = rep.shape

    if label.shape[1] != h or label.shape[2] != w:
        label = F.interpolate(label.unsqueeze(1).float(), size=(h, w), mode='nearest')  # b1hw
    if pred.shape[1] != h or pred.shape[2] != w:
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)  # b2hw

    fea_list = []
    for i in range(2):
        mask1 = (label == i).squeeze(1)
        pred_c = pred[:, i]
        if mask1.sum() > 0:
            thresh1 = np.percentile(pred_c[mask1].cpu().detach().numpy().flatten(), 20)
            mask2 = (pred_c >= thresh1)
        else:
            mask2 = torch.ones_like(mask1)

        if torch.sum(prototype[i]) != 0 and (mask1 * mask2).sum() > 0:
            # pdb.set_trace()
            feat_dis = torch.cosine_similarity(rep[mask1 * mask2], prototype[i])  # 一维
            thresh2 = np.percentile(
                feat_dis.cpu().detach().numpy().flatten(), 20
            )  # 越接近1，相似度越高
            mask3 = (feat_dis > thresh2)  # 一维 取了更好的20%
        else:
            mask3 = torch.ones((mask1 * mask2).sum()).bool()
        fea = rep[mask1 * mask2][mask3]
        fea_list.append(fea)
        keys = (pred_c[mask1 * mask2][mask3])
        dequeue_and_enqueue_prior(feats=fea.detach(),
                                  keys=keys.detach(),  # keys 可以改一下看看
                                  queue=memobank[i],
                                  queue_key=keys_mb[i],
                                  queue_time=queue_time[i],
                                  queue_ptr=queue_prtlis[i],
                                  queue_size=queue_size[i])

    if len(fea_list[0]) > 0 and len(fea_list[1] > 0):
        reco_loss = []
        for i in range(2):
            query_id = torch.randint(len(fea_list[i]), size=(num_queries,))
            query = fea_list[i][query_id]
            key = memobank[1 - i][0]
            key_id = torch.randint(len(key), size=(num_queries * num_negatives,))
            key = key[key_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((prototype[i].unsqueeze(0).repeat(num_queries, 1, 1), key), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
        return (reco_loss[0] + reco_loss[1]) / 2
    else:
        return torch.tensor(0.)


def dv_contra_loss(
        rep_w,  # bs*128*12*12
        rep_s,  # bs*128*12*12
        label_w,  # bs*1*12*12
        label_s,  # bs*1*12*12
        pred_w,  # bs*1*12*12
        pred_s,  # bs*1*12*12
):
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 20
    num_queries = 50
    num_negatives = 256

    rep_w = rep_w.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_s = rep_s.permute(0, 2, 3, 1)  # bs*12*12*128
    b, h, w, c = rep_w.shape
    if label_w.shape[1] != h or label_w.shape[2] != w:
        label_w = F.interpolate(label_w.unsqueeze(1).float(), size=(h, w), mode='nearest')  # b1hw
    if label_s.shape[1] != h or label_s.shape[2] != w:
        label_s = F.interpolate(label_s.unsqueeze(1).float(), size=(h, w), mode='nearest')  # b1hw
    if pred_w.shape[1] != h or pred_w.shape[2] != w:
        pred_w = F.interpolate(pred_w, size=(h, w), mode='bilinear', align_corners=True)
    if pred_s.shape[1] != h or pred_s.shape[2] != w:
        pred_s = F.interpolate(pred_s, size=(h, w), mode='bilinear', align_corners=True)

    fea_w = []
    fea_s = []
    for i in range(2):
        mask1 = ((label_w == i) * (label_s == i)).squeeze(1)
        pred_w_c = pred_w[:, i]
        pred_s_c = pred_s[:, i]
        if mask1.sum() > 0:
            thresh1 = np.percentile(
                torch.cat((pred_w_c[mask1], pred_s_c[mask1]), dim=0).cpu().detach().numpy().flatten(), 50)
            mask2 = ((pred_w_c >= thresh1) * (pred_s_c >= thresh1))
        else:
            mask2 = torch.ones_like(mask1)

        fea_w.append(rep_w[mask1 * mask2])
        fea_s.append(rep_s[mask1 * mask2])

    if len(fea_w[0]) > 0 and len(fea_w[1] > 0):
        reco_loss = []
        for i in range(2):
            query_id = torch.randint(len(fea_w[i]), size=(num_queries,))
            query = fea_s[i][query_id]
            positive = fea_w[i][query_id]
            negatives = torch.cat((fea_w[1 - i], fea_s[1 - i]), dim=0)
            key_id = torch.randint(len(negatives), size=(num_queries * num_negatives,))
            negatives = negatives[key_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((positive.unsqueeze(1), negatives), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
        return (reco_loss[0] + reco_loss[1]) / 2
    else:
        return torch.tensor(0.)

    # if len(fea_w[0]) > 0 and len(fea_w[1] > 0):
    #     reco_loss = []
    #     for i in range(2):
    #         query_id = torch.randint(len(fea_w[i]), size=(num_queries,))
    #         z1 = fea_w[i][query_id]
    #         z2 = fea_s[i][query_id]
    #         negatives = torch.cat((fea_w[1 - i], fea_s[1 - i]), dim=0)
    #         key_id = torch.randint(len(negatives), size=(num_queries * num_negatives,))
    #         negatives = negatives[key_id].reshape(num_queries, num_negatives, c)
    #         z1_feat = torch.cat((z2.unsqueeze(1), negatives), dim=1)
    #         z2_feat = torch.cat((z1.unsqueeze(1), negatives), dim=1)
    #         seg_logits1 = torch.cosine_similarity(z1.unsqueeze(1), z1_feat.detach(), dim=2)
    #         seg_logits2 = torch.cosine_similarity(z2.unsqueeze(1), z2_feat.detach(), dim=2)
    #         reco_loss.append(
    #             F.cross_entropy(seg_logits1 / temp, torch.zeros(num_queries).long().cuda()) +
    #             F.cross_entropy(seg_logits2 / temp, torch.zeros(num_queries).long().cuda()))
    #
    #     return (reco_loss[0] + reco_loss[1]) / 2
    # else:
    #     return torch.tensor(0.)

    # if len(fea_w[0]) > 0 and len(fea_w[1] > 0):
    #     reco_loss = []
    #     for i in range(2):
    #         z_id = torch.randint(len(fea_w[i]), size=(num_queries,))
    #         z = torch.stack((fea_w[i][z_id], fea_s[i][z_id]), dim=1)  # 50*2*256
    #
    #         zsub_id = torch.randint(len(fea_w[1 - i]), size=(num_queries * num_negatives,))
    #         zsub = torch.cat((fea_w[1 - i][zsub_id].reshape(num_queries, num_negatives, c),
    #                           fea_s[1 - i][zsub_id].reshape(num_queries, num_negatives, c)), dim=1)  # 50*(2*256)*c
    #
    #         pair_logits = torch.cosine_similarity(z[:, 0], z[:, 1], dim=1).unsqueeze(-1)  # 50*1
    #         neg_logits = torch.cosine_similarity(z.unsqueeze(2), zsub.unsqueeze(1), dim=3).view(num_queries,
    #                                                                                                      -1)  # 50*(2*512)
    #         seg_logits = torch.cat((pair_logits, neg_logits), dim=1)
    #         reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
    #     return (reco_loss[0] + reco_loss[1]) / 2
    # else:
    #     return torch.tensor(0.)


def dv_ln_contra_loss(
        rep_w,  # bs*128*12*12
        rep_s,  # bs*128*12*12
        label_w,  # bs*1*12*12
        label_s,  # bs*1*12*12
        pred_w,  # bs*1*12*12
        pred_s,  # bs*1*12*12
        rep_l,
        label_l,
        pred_l
):
    thre = 0.5  # 0.5是为了分类，alpha_t其实是不确定性
    temp = 0.5
    alpha_t = 20
    num_queries = 50
    num_negatives = 256

    rep_w = rep_w.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_s = rep_s.permute(0, 2, 3, 1)  # bs*12*12*128
    rep_l = rep_l.permute(0, 2, 3, 1)

    b, h, w, c = rep_w.shape
    if label_w.shape[1] != h or label_w.shape[2] != w:
        label_w = F.interpolate(label_w.unsqueeze(1).float(), size=(h, w), mode='nearest')  # b1hw
    if label_s.shape[1] != h or label_s.shape[2] != w:
        label_s = F.interpolate(label_s.unsqueeze(1).float(), size=(h, w), mode='nearest')  # b1hw
    if label_l.shape[1] != h or label_l.shape[2] != w:
        label_l = F.interpolate(label_l.unsqueeze(1).float(), size=(h, w), mode='nearest')  # b1hw
    if pred_w.shape[1] != h or pred_w.shape[2] != w:
        pred_w = F.interpolate(pred_w, size=(h, w), mode='bilinear', align_corners=True)
    if pred_s.shape[1] != h or pred_s.shape[2] != w:
        pred_s = F.interpolate(pred_s, size=(h, w), mode='bilinear', align_corners=True)
    if pred_l.shape[1] != h or pred_l.shape[2] != w:
        pred_l = F.interpolate(pred_l, size=(h, w), mode='bilinear', align_corners=True)

    fea_w = []
    fea_s = []
    for i in range(2):
        mask1 = ((label_w == i) * (label_s == i)).squeeze(1)
        pred_w_c = pred_w[:, i]
        pred_s_c = pred_s[:, i]
        if mask1.sum() > 0:
            thresh1 = np.percentile(
                torch.cat((pred_w_c[mask1], pred_s_c[mask1]), dim=0).cpu().detach().numpy().flatten(), 50)
            mask2 = ((pred_w_c >= thresh1) * (pred_s_c >= thresh1))
        else:
            mask2 = torch.ones_like(mask1)

        fea_w.append(rep_w[mask1 * mask2])
        fea_s.append(rep_s[mask1 * mask2])

    fea_l = []
    for i in range(2):
        mask1 = (label_l == i).squeeze(1)
        pred_l_c = pred_l[:, i]
        if mask1.sum() > 0:
            thresh1 = np.percentile(
                (pred_l_c[mask1]).cpu().detach().numpy().flatten(), 50)
            mask2 = pred_l_c >= thresh1
        else:
            mask2 = torch.ones_like(mask1)
        fea_l.append(rep_l[mask1 * mask2])

    if len(fea_w[0]) > 0 and len(fea_w[1] > 0):
        reco_loss = []
        for i in range(2):
            query_id = torch.randint(len(fea_w[i]), size=(num_queries,))
            query = fea_s[i][query_id]
            positive = fea_w[i][query_id]
            # negatives = torch.cat((fea_w[1 - i], fea_s[1 - i]), dim=0)
            negatives = fea_l[1 - i]
            key_id = torch.randint(len(negatives), size=(num_queries * num_negatives,))
            negatives = negatives[key_id].reshape(num_queries, num_negatives, c)
            all_feat = torch.cat((positive.unsqueeze(1), negatives), dim=1)
            seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat.detach(), dim=2)
            reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
        return (reco_loss[0] + reco_loss[1]) / 2
    else:
        return torch.tensor(0.)

    # if len(fea_w[0]) > 0 and len(fea_w[1] > 0):
    #     reco_loss = []
    #     for i in range(2):
    #         query_id = torch.randint(len(fea_w[i]), size=(num_queries,))
    #         z1 = fea_w[i][query_id]
    #         z2 = fea_s[i][query_id]
    #         negatives = torch.cat((fea_w[1 - i], fea_s[1 - i]), dim=0)
    #         key_id = torch.randint(len(negatives), size=(num_queries * num_negatives,))
    #         negatives = negatives[key_id].reshape(num_queries, num_negatives, c)
    #         z1_feat = torch.cat((z2.unsqueeze(1), negatives), dim=1)
    #         z2_feat = torch.cat((z1.unsqueeze(1), negatives), dim=1)
    #         seg_logits1 = torch.cosine_similarity(z1.unsqueeze(1), z1_feat.detach(), dim=2)
    #         seg_logits2 = torch.cosine_similarity(z2.unsqueeze(1), z2_feat.detach(), dim=2)
    #         reco_loss.append(
    #             F.cross_entropy(seg_logits1 / temp, torch.zeros(num_queries).long().cuda()) +
    #             F.cross_entropy(seg_logits2 / temp, torch.zeros(num_queries).long().cuda()))
    #
    #     return (reco_loss[0] + reco_loss[1]) / 2
    # else:
    #     return torch.tensor(0.)

    # if len(fea_w[0]) > 0 and len(fea_w[1] > 0):
    #     reco_loss = []
    #     for i in range(2):
    #         z_id = torch.randint(len(fea_w[i]), size=(num_queries,))
    #         z = torch.stack((fea_w[i][z_id], fea_s[i][z_id]), dim=1)  # 50*2*256
    #
    #         zsub_id = torch.randint(len(fea_w[1 - i]), size=(num_queries * num_negatives,))
    #         zsub = torch.cat((fea_w[1 - i][zsub_id].reshape(num_queries, num_negatives, c),
    #                           fea_s[1 - i][zsub_id].reshape(num_queries, num_negatives, c)), dim=1)  # 50*(2*256)*c
    #
    #         pair_logits = torch.cosine_similarity(z[:, 0], z[:, 1], dim=1).unsqueeze(-1)  # 50*1
    #         neg_logits = torch.cosine_similarity(z.unsqueeze(2), zsub.unsqueeze(1), dim=3).view(num_queries,
    #                                                                                                      -1)  # 50*(2*512)
    #         seg_logits = torch.cat((pair_logits, neg_logits), dim=1)
    #         reco_loss.append(F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda()))
    #     return (reco_loss[0] + reco_loss[1]) / 2
    # else:
    #     return torch.tensor(0.)


def getPrototype(features, mask, confidence, c):
    '''
    feature:bs*d*h/16*w/16
    mask:bs*h*w
    class_confidence:bs*c*h*w
    '''
    mask_onehot = F.one_hot(mask.long(), c).permute(0, 3, 1, 2).float()  # bs*c*h/16*w/16
    proto = torch.stack([torch.sum(features * mask_onehot[:, i:i + 1] * confidence[:, i:i + 1], dim=(2, 3)) / (
            (mask_onehot[:, i:i + 1] * confidence[:, i:i + 1]).sum(dim=(2, 3)) + 1e-5) for i in range(c)],
                        dim=1)  # bs x C
    return proto


def qkv_loss(rep_l, rep_ul, rep_ul_p, gt_l, pred_ul, pred_ul_p, r=0.05, scale=2, wa=False):
    """
    f_label=query bs*256*24*24
    f_unlabel=key bs*256*24*24
    f_unlabel_per=value bs*256*24*24
    pred_ul, pred_ul_p bs*2*24*24
    gt_l bs*24*24
    label有噪声怎么办？pred是不是也要detach一下？
    """
    bl, d, h, w = rep_l.shape
    bu, c, _, _ = pred_ul.shape

    pro_l = getPrototype(rep_l, gt_l, torch.ones(bl, c, h, w).cuda(), c)
    pro_ul = getPrototype(rep_ul, torch.argmax(pred_ul, dim=1), pred_ul, c)
    pro_ul_p = getPrototype(rep_ul_p, torch.argmax(pred_ul_p, dim=1), pred_ul_p, c)

    if wa:
        # att = (pro_l[:, 1] @ pro_ul[:, 1].transpose(0, 1)).detach()
        att = torch.cosine_similarity(pro_l[:, 1].unsqueeze(1), pro_ul[:, 1].unsqueeze(0), dim=-1)
        att_soft = torch.softmax(att / r, dim=-1).detach()
        att_soft_con = torch.softmax(att.transpose(0, 1) / r, dim=-1).detach()  # bu,bl 因为是unlabeled去选择和自己相似的labeled
        # pdb.set_trace()
    else:
        att_soft = torch.ones((bl, bu)).cuda() / bu
        att_soft_con = torch.ones((bu, bl)).cuda() / bl
        # pdb.set_trace()

    # one
    rep_l_ = rep_l.permute(0, 2, 3, 1).view(bl, 1, 1, h, w, d)
    ugl = torch.cosine_similarity(rep_l_, pro_ul.view(1, bu, c, 1, 1, d), dim=-1)  # bl,bu,c,h,w [-1,1]
    upgl = torch.cosine_similarity(rep_l_, pro_ul_p.view(1, bu, c, 1, 1, d), dim=-1)

    torch.use_deterministic_algorithms(False)
    loss_ugl = (att_soft * torch.stack(
        [F.cross_entropy(ugl[:, i] * scale, gt_l.long(),
                         reduction='none').mean(dim=(1, 2)) for i in range(bu)], dim=1)).sum(1).mean()
    loss_upgl = (att_soft * torch.stack(
        [F.cross_entropy(upgl[:, i] * scale, gt_l.long(),
                         reduction='none').mean(dim=(1, 2)) for i in range(bu)], dim=1)).sum(1).mean()  # bl*bu
    loss1 = (loss_ugl + loss_upgl) / 2
    torch.use_deterministic_algorithms(True)

    # two
    rep_ul_ = rep_ul.permute(0, 2, 3, 1).view(bu, 1, 1, h, w, d)
    lgu = torch.cosine_similarity(rep_ul_, pro_l.view(1, bl, c, 1, 1, d), dim=-1)  # bu*bl*c*h*w
    rep_ul_p_ = rep_ul_p.permute(0, 2, 3, 1).view(bu, 1, 1, h, w, d)
    lgup = torch.cosine_similarity(rep_ul_p_, pro_l.view(1, bl, c, 1, 1, d), dim=-1)
    # pdb.set_trace()

    loss2 = (att_soft_con * torch.stack(
        [F.mse_loss(torch.softmax(lgu[:, i] * scale, dim=1),
                    torch.softmax(lgup[:, i] * scale, dim=1),
                    reduction='none').mean(dim=(1, 2, 3)) for i in range(bl)], dim=1)).sum(1).mean()

    #
    # for i in range(bl):
    #     cv2.imwrite("vis/{}_gt.png".format(i), (gt_l[i] * 255).detach().cpu().numpy())
    #     for j in range(bu):
    #         cv2.imwrite('vis/{}_{}.png'.format(i, j),
    #                     (torch.softmax(F.interpolate(ugl[i], size=gt_l.shape[-2:], mode='bilinear') * 2, dim=1)[
    #                          j, 1] * 255).detach().cpu().numpy())
    # pdb.set_trace()
    return loss1, loss2, att_soft, ugl


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
