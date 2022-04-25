from sahi.predict import predict
import argparse
import os
from utils import mAP, pickle_to_text, yolo_to_txt, read_file_to_tensor
import wandb
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
    parser.add_argument('--source_image_dir', default="../SeaDroneSee_challenge/data/images_drones/", type=str)
    parser.add_argument('-m', '--mode', default='sparse', type=str)
    parser.add_argument('--fmaps_path', default='./actual_baseline_1_images_drones.npy', type=str)
    parser.add_argument('-n', '--name', default='exp', type=str)
    args = parser.parse_args()
    model_confidence_threshold_arr = [0.25, 0.5, 0.75]
    slice_arr = [512, 1024]
    overlap_ratio_arr = [0.0, 0.2]
    postprocess_type_arr = ['NMS', 'NMM', 'GREEDYNMM']
    postprocess_match_metric_arr = ['IOU', 'IOS']
    postprocess_match_threshold_arr = [0.25, 0.5, 0.75]
    postprocess_class_agnostic_arr = [True, False]
    cnt = 1
    for mct in model_confidence_threshold_arr:
        for s in slice_arr:
            for or_ in overlap_ratio_arr:
                for pt in postprocess_type_arr:
                    for pmm in postprocess_match_metric_arr:
                        for pmt in postprocess_match_threshold_arr:
                            for pca in postprocess_class_agnostic_arr:
                                cfg = {
                                    'model_confidence_threshold_arr': mct,
                                    'slice_height_arr': s,
                                    'slice_width_arr': s,
                                    'overlap_height_ratio_arr': or_,
                                    'overlap_width_ratio_arr': or_,
                                    'postprocess_type_arr': pt,
                                    'postprocess_match_metric_arr': pmm,
                                    'postprocess_match_threshold_arr': pmt,
                                    'postprocess_class_agnostic_arr': pca
                                }
                                run = wandb.init(
                                    project="sahi",
                                    name=datetime.now().strftime("%b%d_%H:%M:%S"),
                                    entity="cyr1ll",
                                    group='grid-search',
                                    config=cfg
                                )
                                predict(
                                    model_type=model_type,
                                    model_path=args.model_path,
                                    model_device=model_device,
                                    model_confidence_threshold=mct,
                                    source=args.source_image_dir,
                                    slice_height=s,
                                    slice_width=s,
                                    overlap_height_ratio=or_,
                                    overlap_width_ratio=or_,
                                    export_pickle=True,
                                    export_visual=True,
                                    mode='standard',
                                    image_size=768,
                                    postprocess_type=pt,
                                    postprocess_match_metric=pmm,
                                    postprocess_match_threshold=pmt,
                                    postprocess_class_agnostic=pca,
                                    name=f"exp{cnt}"
                                )

                                pickle_in = f"./runs/predict/exp{cnt}/pickles/"
                                pickle_to_text_dir = "./pickles_to_text/"
                                print("Pickles at: ", pickle_in)

                                gt_in = './ground_truth/gt_images_drones/'
                                images_path = '../SeaDroneSee_challenge/data/images_drones/'
                                gt_to_text_dir = './ground_truth/gt_images_drones_txt/'
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
                                wandb.log(res)
                                print(res)
                                cnt += 1
                                run.finish()
