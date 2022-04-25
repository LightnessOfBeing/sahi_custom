from sahi.predict import predict
import argparse
import os
from utils import mAP, pickle_to_text, yolo_to_txt, read_file_to_tensor
import json

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
    predict(
        model_type=model_type,
        model_path=args.model_path,
        model_device=model_device,
        model_confidence_threshold=model_confidence_threshold,
        source=args.source_image_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        export_pickle=True,
        export_visual=True,
        #no_standard_prediction=True, 
        #no_sliced_prediction=True,
        mode=args.mode,
        fmaps_path=args.fmaps_path,
        image_size=768,
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.15,
        postprocess_class_agnostic=True,
        postprocess_type="NMS",
        name="exp"
    )

    pickle_in = "./runs/predict/exp/pickles/"
    pickle_to_text_dir = "./pickles_to_text/"
    print("Pickles at: ", pickle_in)

    #ground truth
    gt_in = './ground_truth/gt_images_drones/'
    images_path = '../SeaDroneSee_challenge/data/images_drones/'
    gt_to_text_dir = './ground_truth/gt_images_drones_txt/'
    print("Ground truth at: ", gt_to_text_dir) 

    pickle_to_text(pickle_in, pickle_to_text_dir)
    yolo_to_txt(gt_in, gt_to_text_dir, images_path)

    all_labels = []
    all_detections = []
    print(os.listdir(gt_to_text_dir))
    for filename in os.listdir(gt_to_text_dir):
        labels = read_file_to_tensor(gt_to_text_dir, filename, n_fields=5)
        detections = read_file_to_tensor(pickle_to_text_dir, filename, n_fields=6)

        all_labels.append(labels)
        all_detections.append(detections)

    res = mAP(all_detections, all_labels)

    with open('./results/result.json' , 'w') as outfile:
        json.dump(res, outfile)
    print(res)