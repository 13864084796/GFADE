import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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


def main(args):
    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    # 加载 FaceForensics++ 数据集
    video_list, target_list = init_ff_tsne(dataset='all', phase='test')

    # 初始化 t-SNE 特征和标签列表
    features = []
    labels = []

    for filename, label in tqdm(zip(video_list, target_list), total=len(video_list)):
        try:
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)

            with torch.no_grad():
                img = torch.tensor(face_list).to(device).float() / 255
                # 获取模型的中间层特征
                features_layer = model.net.extract_features(img)  # 假设 EfficientNet 的 extract_features 方法
                features_layer = features_layer.mean(dim=[2, 3])  # 全局平均池化，得到每个图像的特征
                # 计算每个视频的特征均值，以确保每个视频仅有一个特征表示
                video_feature = features_layer.mean(dim=0).cpu().numpy()

                features.append(video_feature)
                labels.append(label)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # 将所有特征和标签转换为 numpy 数组
    features = np.array(features)
    labels = np.array(labels)

    # 使用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(5, len(features) - 1))  # 设置 perplexity 小于样本数
    features_2d = tsne.fit_transform(features)

    # 设置颜色
    colors = {0: 'blue', 1: 'red', 2: 'orange', 3: 'green', 4: 'yellow'}
    label_names = {0: "Real", 1: "Deepfakes", 2: "Face2Face", 3: "FaceSwap", 4: "NeuralTextures"}

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 10))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], color=colors[label], label=label_names[label],
                    alpha=0.6)
    plt.legend()
    plt.title("t-SNE visualization of FaceForensics++ features")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")

    # 保存 t-SNE 图
    plt.savefig("tsne_visualization.png")
    plt.show()


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
