import cv2
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import argparse
import imagesize

def normalize(features):
    return (features - np.min(features)) / (np.max(features) - np.min(features))

def get_mask(array, t=0.5):
    arr = array.copy()
    arr[arr > t] = 1
    arr[arr <= t] = 0
    return arr

def prepare(layer_name, fmaps_folder, out_name, mode, alignment_x=0.075, alignment_y=0.075):
    folders = sorted(list(filter(lambda x: "png" not in x and x != "labels", os.listdir(fmaps_folder))))
    file_names = sorted([os.path.join(os.path.join(os.path.join(fmaps_folder, folder), layer_name)) for folder in folders])
   
    xs_arr = []
    ys_arr = []
    xe_arr = []
    ye_arr = []
    filename_arr = []
    
    final_arr = []
    for folder, file in tqdm(zip(folders, file_names)):
        filename_arr.append(f"{folder}.png")
        #image = cv2.imread(os.path.join(fmaps_folder, f"{folder}.png")) # todo substitute with image file
        #orig_height, orig_width, _ = image.shape
        orig_width, orig_height = imagesize.get(os.path.join(fmaps_folder, f"{folder}.png"))

        features = np.load(file).mean(axis=0)
        blurred_features = cv2.blur(features, (3, 3))
        normalized_blurred_features = normalize(blurred_features)
        blurred_features_mask = get_mask(normalized_blurred_features)

        if mode == "naive":
            y, x, h, w = cv2.boundingRect(np.argwhere(blurred_features_mask == 1))
            final_mask = np.zeros_like(blurred_features_mask)
            final_mask[y: y + h, x: x + w] = 1
            x_e = x + w
            y_e = y + h
            f_height, f_width = final_mask.shape
            x = max(0, x - alignment_x * f_width)
            y = max(0, y - alignment_y * f_height)
            x_e = min(f_width, x_e + alignment_x * f_width)
            y_e = min(f_height, y_e + alignment_y * f_height)
            w = x_e - x
            h = y_e - y
            mult_height = orig_height / f_height
            mult_width = orig_width / f_width
            imxs = int(x * mult_width)
            imys = int(y * mult_height)
            imxe = int(x * mult_width + w * mult_width) 
            imye = int(y * mult_height + h * mult_height)
            xs_arr.append(imxs)
            ys_arr.append(imys)
            xe_arr.append(imxe)
            ye_arr.append(imye)
        elif mode == "sparse":
            arr = torch.Tensor(blurred_features_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            final_mask_interpolated = torch.nn.functional.interpolate(arr, size=(orig_height, orig_width)).numpy()[0][0]
            final_arr.append(final_mask_interpolated)
        else:
            raise ValueError("Not implemented!")
        
    if mode == "naive":
        df = pd.DataFrame({'file': filename_arr, 'xs': xs_arr, 'ys': ys_arr, 'xe': xe_arr, 'ye': ye_arr})
        df.to_csv(f"./{out_name}.csv", index=False)
    elif mode == "sparse":
        final_arr = np.array(final_arr)
        np.save(f"./{out_name}.npy", final_arr, allow_pickle=True)
    else:
        raise ValueError("Not implemented!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom SAHI preparation')
    parser.add_argument('--layer_name', default="stage16_Concat_features.npy", type=str, help='layer name from which feature are coming')
    parser.add_argument('--fmaps_folder', default="../SeaDroneSee_challenge/runs/detect/baseline_val_full3/", type=str, help='folder which contains computed feature maps')
    parser.add_argument('--out_name', default="baseline_val_full", type=str, help='output name')
    parser.add_argument('--mode', default='sparse', type=str, help='way of computing feature maps, two options are available: naive and sparse.')
    args = parser.parse_args()
    prepare(layer_name=args.layer_name, fmaps_folder=args.fmaps_folder, out_name=args.out_name, mode=args.mode)
