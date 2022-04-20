from sahi.predict import predict

model_type = "yolov5"
model_path = "../SeaDroneSee_challenge/runs/train/actual_baseline_1/weights/best.pt"
model_device = 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 512
slice_width = 512
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0

source_image_dir = "../SeaDroneSee_challenge/data/images_drones/"

if __name__ == "__main__":
    predict(
        model_type=model_type,
        model_path=model_path,
        model_device=model_device,
        model_confidence_threshold=model_confidence_threshold,
        source=source_image_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        export_pickle=True,
        export_visual=True,
        mode="sparse",
        fmaps_path="/home/kirill.vishniakov/research/SeaDroneSee_challenge/notebooks/actual_baseline_1_images_drones.npy",
        #fmaps_layer="stage16_Concat_features.npy"
    )
    # predict(
    #     model_type=model_type,
    #     model_path=model_path,
    #     model_device=model_device,
    #     model_confidence_threshold=model_confidence_threshold,
    #     source=source_image_dir,
    #     slice_height=slice_height,
    #     slice_width=slice_width,
    #     overlap_height_ratio=overlap_height_ratio,
    #     overlap_width_ratio=overlap_width_ratio,
    #     export_pickle=True,
    #     export_visual=True,
    #     mode="naive",
    #     fmaps_path="/home/kirill.vishniakov/research/SeaDroneSee_challenge/runs/detect/actual_baseline_1",
    #     fmaps_layer="stage16_Concat_features.npy"
    # )