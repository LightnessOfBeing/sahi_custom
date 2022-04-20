from sahi.predict import predict
import argparse

model_type = "yolov5"
model_device = 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 512
slice_width = 512
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom SAHI')
    parser.add_argument('--model_path', default="../SeaDroneSee_challenge/runs/train/actual_baseline_1/weights/best.pt", type=str)
    parser.add_argument('--source_image_dir', default="../SeaDroneSee_challenge/data/images_drones/", type=str)
    parser.add_argument('--mode', default='sparse', type=str)
    parser.add_argument('--fmaps_path', default='../SeaDroneSee_challenge/notebooks/actual_baseline_1_images_drones.npy', type=str)
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
        mode="sparse",
        fmaps_path=args.fmaps_path,
    )