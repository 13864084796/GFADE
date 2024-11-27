import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_grad_cam_heatmap(model, input_tensor, target_layer, device):
    cam = GradCAM(model=model, target_layers=[target_layer])  # Removed 'use_cuda' as it's deprecated
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])
    del cam  # Explicitly delete the GradCAM object to release resources
    return grayscale_cam[0]


def main(args):
    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list, target_list = init_ffiw()
    elif args.dataset == 'FF':
        video_list, target_list = init_ff()
    elif args.dataset == 'FF_c40':
        video_list, target_list = init_ff_c40()
    elif args.dataset == 'FF_c23':
        video_list, target_list = init_ff_c23()
    elif args.dataset == 'DFD':
        video_list, target_list = init_dfd()
    elif args.dataset == 'DFDC':
        video_list, target_list = init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list, target_list = init_dfdcp()
    elif args.dataset == 'CDF':
        video_list, target_list = init_cdf()
    else:
        NotImplementedError

    for filename in tqdm(video_list):
        try:
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)

            with torch.no_grad():
                img_tensor = torch.tensor(face_list).to(device).float() / 255
                pred = model(img_tensor)[0].softmax(1)[:, 1]

            # Grad-CAM part
            heatmap_results = []
            for i in range(len(img_tensor)):
                img = img_tensor[i].unsqueeze(0)  # Adding batch dimension
                target_layer = model.net._blocks[-1]  # Choosing last convolutional layer in EfficientNet
                grayscale_cam = generate_grad_cam_heatmap(model, img, target_layer, device)
                original_image = face_list[i].transpose(1, 2, 0).astype(
                    np.float32) / 255.0  # Convert CHW to HWC and normalize to [0, 1]
                heatmap = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
                heatmap_results.append((original_image, heatmap))

            # Plotting original and heatmap side by side
            fig, axs = plt.subplots(len(face_list), 2, figsize=(8, len(face_list) * 4))
            for i, (original, heatmap) in enumerate(heatmap_results):
                axs[i, 0].imshow(original)
                axs[i, 0].axis('off')
                axs[i, 0].set_title("DF")

                axs[i, 1].imshow(heatmap)
                axs[i, 1].axis('off')
                axs[i, 1].set_title("Grad-CAM Heatmap(Ours)")

            plt.tight_layout()
            plt.savefig(f'DF/gradcam_{args.dataset}_{os.path.basename(filename)}.png')
            plt.close()

        except Exception as e:
            print(e)


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_name', type=str)
    parser.add_argument('-d', dest='dataset', type=str)
    parser.add_argument('-n', dest='n_frames', default=32, type=int)
    args = parser.parse_args()

    main(args)
