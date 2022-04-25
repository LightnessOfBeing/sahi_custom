import shutil
import torch
import numpy as np
import shutil
import imagesize
import os
import pandas as pd
from torch import tensor, cat
from typing import List
from pathlib import Path

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def mAP(preds: List[torch.Tensor], labels: List[torch.Tensor]): 
    """
    PREDICTIONS AND LABELS FOR THE WHOLE DATASET
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        COORDS ARE FLOAT, ACTUAL IMAGE COORDINATES (NOT RESIZED, FULL RESOLUTION)
        labels (Array[M, 5]), class, x1, y1, x2, y2 
        top left bottom right
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []                                                                                                             
    #for (pred, gt) in tqdm(zip(preds, labels), desc='Calcuating TP', total=len(preds)):
    for (pred, gt) in zip(preds, labels):
       # print(process_batch(pred, gt, iouv))                                                                              
      #  correct, iou_scores = process_batch(pred, gt, iouv)
        correct = process_batch(pred, gt, iouv)
        true_classes = gt[:, 0].tolist()
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), true_classes))  # (correct, conf, pcls, tcls)

    # true_classes = labels[:, 0].tolist()
    # stats = [(correct.cpu(), preds[:, 4].cpu(), preds[:, 5].cpu(), true_classes)]  # (correct, conf, pcls, tcls)
    #print(stats.shape)

    # Compute metrics
    # n = num_det
    # m = num_gt_boxes
    # k = num_batches
    # stats: [k] -> [(is_correct: [n x num_iou_thresholds; bool], confidence: [n], predicted_class: [n], true_class: [m]), ...]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy 
    # stats: [4] -> (is_correct: [n * k x num_iou_thresholds; bool], confidence: [n * k], predicted_class: [n * k], true_class: [m * k])

    names = ['swimmer', 'floater', 'boat', 'swimmer_on_boat', 'floater_on_boat', 'life_jacket']
    print(len(stats))
    if len(stats) and stats[0].any():
        # p, r, f1, ap - np.array((num_classes, ))
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=Path(""), names={i: name for i, name in enumerate(names)})
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        #mIoU = torch.mean(torch.cat(iou_scores))

    return {
        "map": map,
        "map50": map50,
        "mp": mp,
        "mr": mr,
        #"mIoU": mIoU,        
    }

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')

def yolo_to_txt(folder_in, folder_out, folder_images):

    if os.path.isdir(folder_out): 
        shutil.rmtree(folder_out)
    os.mkdir(folder_out)

    for filename in os.listdir(folder_in):
        img_name = os.path.splitext(filename)[0]
        w_img, h_img = imagesize.get(folder_images+img_name+".png")

        # open yolo and new file
        with open(folder_in+img_name+".txt", 'r') as f:
            with open(folder_out+img_name+".txt", 'w') as f_new:

                # read all lines in yolo
                lines = f.readlines()
                
                # write all lines in new
                for line in lines:
                    line_list = line.strip().split(' ')

                    class_obj = line_list[0]
                    cx = float(line_list[1])
                    cy = float(line_list[2])
                    w = float(line_list[3])
                    h = float(line_list[4]) 
                    
                    # conversion
                    minx = int((cx - w/2) * w_img)
                    miny = int((cy - h/2) * h_img) 

                    maxy = int((cy + h/2) * h_img)
                    maxx = int ((cx + w/2) * w_img)
                    
                    #if img_name == "1860":
                        #print(filename, minx, miny, maxx, maxy)

                    #number.txt <class_name> <left> <top> <right> <bottom>
                    f_new.write(str(class_obj) + " ") 
                    f_new.write(str(minx) + " ") 
                    f_new.write(str(miny) + " ") 
                    f_new.write(str(maxx) + " ") 
                    f_new.write(str(maxy) + " ") 
                    f_new.write("\n")

            f_new.close()              
        f.close()

def pickle_to_text(folder_in, folder_out):
    if os.path.isdir(folder_out): 
        shutil.rmtree(folder_out)
    os.mkdir(folder_out)

    for filename in os.listdir(folder_in):
        with open(folder_out+os.path.splitext(filename)[0]+".txt", 'w') as f:
            object = pd.read_pickle(folder_in+filename)

            for i in range(0, len(object)):
                f.write(str(object[i].category.id) + " ") 
                f.write(str(object[i].bbox.minx) + " " + str(object[i].bbox.miny) + " " +str(object[i].bbox.maxx) + " " + str(object[i].bbox.maxy)  + " ") 
                f.write(str(object[i].score.value)) 
                f.write("\n")

        f.close()

def read_file_to_tensor(path, filename, n_fields):
    with open(path + filename) as f:
        t_lines = torch.empty(0, n_fields)

        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split(' ')
            class_obj = int(line_list[0])
            min_x = float(line_list[1])
            min_y = float(line_list[2])
            max_x = float(line_list[3])
            max_y = float(line_list[4]) 
            

            if n_fields == 5:
                #labels (Array[M, 5]), class, x1, y1, x2, y2 
                new_line = tensor([[class_obj, min_x, min_y, max_x, max_y]])
            else:
                #detections (Array[N, 6]), x1, y1, x2, y2, conf, class
                score = float(line_list[5]) 
                new_line = tensor([[min_x, min_y, max_x, max_y, score, class_obj]])

            #print("New line:", new_line)
            t_lines = cat([new_line[:-1], t_lines, new_line[-1:]]) #try 1
            #print("List", t_lines)

    return t_lines
