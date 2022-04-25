from sahi.predict import predict
import argparse
import os
from utils import mAP, pickle_to_text, yolo_to_txt, read_file_to_tensor
import json
import wandb
import time
from datetime import datetime

model_type = "yolov5"
model_device = 'cuda:0'
model_confidence_threshold = 0.35

slice_height = 1024
slice_width = 1024
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom SAHI')
    parser.add_argument('--model_path', default="../SeaDroneSee_challenge/runs/train/actual_baseline_1/weights/best.pt", type=str)
    parser.add_argument('--source_image_dir', default="../datasets/SeaDroneSee/val/images", type=str)
    parser.add_argument('-m', '--mode', default='standard', type=str)
    parser.add_argument('--fmaps_path', default='./baseline_val_full.npy', type=str)
    parser.add_argument('--fmaps_layer', default='stage16_Concat_features.npy', type=str)
    parser.add_argument('-n', '--name', default='exp', type=str)
    args = parser.parse_args()
    cnt = 1
    for slice_size in [512, 768, 1024]:
        name = str(datetime.now().strftime("%b%d_%H:%M:%S"))
        run = wandb.init(
                project="sahi",
                name=name,
                entity="cyr1ll",
                group='metrics',
                tags=[args.mode],
                config={'slice_size': slice_size, 'mode': args.mode},
            )
        start = time.time()
        total_num_slices = predict(
            model_type=model_type,
            model_path=args.model_path,
            model_device=model_device,
            model_confidence_threshold=model_confidence_threshold,
            source=args.source_image_dir,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            export_pickle=True,
            export_visual=True,
            mode=args.mode,
            fmaps_path=args.fmaps_path,
            fmaps_layer=args.fmaps_layer,
            image_size=768,
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.15,
            postprocess_class_agnostic=True,
            postprocess_type="NMS",
            name=f"metrics{name}"
        )
        elapsed = (time.time() - start)
        actual_time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))

        pickle_in = f'./runs/predict/metrics{name}/pickles/'
        pickle_to_text_dir = "./pickles_to_text/"
        print("Pickles at: ", pickle_in)

        #ground truth
        gt_in = './datasets/SeaDroneSee/val/labels'
        images_path = '../datasets/SeaDroneSee/val/images'
        gt_to_text_dir = './ground_truth/gt_val_full/'
        print("Ground truth at: ", gt_to_text_dir) 

        pickle_to_text(pickle_in, pickle_to_text_dir)
        yolo_to_txt(gt_in, gt_to_text_dir, images_path)

        all_labels = []
        all_detections = []
        for filename in os.listdir(gt_to_text_dir):
            labels = read_file_to_tensor(gt_to_text_dir, filename, n_fields=5)
            detections = read_file_to_tensor(pickle_to_text_dir, filename, n_fields=6)

            all_labels.append(labels)
            all_detections.append(detections)

        res = mAP(all_detections, all_labels)
        res['total_num_slices'] = total_num_slices
        res['time'] = actual_time
        wandb.log(res)
        cnt += 1
        run.finish()

    