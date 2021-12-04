import os
import time
import datetime
import random
import glob
import cv2
import csv
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageShow, ImageFont
from tqdm.notebook import tqdm as tqdm
from operator import itemgetter
from tabulate import tabulate
from utils.engine import evaluate

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

DATASET = 'data'
TESTING = 'test_imgs'
REAL_DATASET = 'real_dataset'

FALLEN_COLOR = (255, 0, 0) # Red
NO_FALLEN_COLOR = (0, 255, 0) #Green
TEXT_COLOR = (255, 255, 255) # White
BOX_COLOR = (0, 0, 0) #Black

files = glob.glob('data/**/*.txt', recursive = True) #find all *.txt files

target_txt = "train_dataset.txt"
real_train_txt = "real_train.txt"
real_valid_txt = "real_valid.txt"
test_txt = "test_set.txt"
category_id_to_name = {1: 'no fallen', 2: 'fallen'}
original_stdout = sys.stdout

def filename(file_format, name=None):
    x = datetime.datetime.now()
    timestamp = str(x)
    timestamp = timestamp.replace("-","_")
    timestamp = timestamp.replace(":","_")
    date = timestamp.split(" ")
    time = date[1]
    time = time.split("_")
    if file_format.startswith("."):
        file_format = file_format.replace(".","")
    filename = date[0] + time[0] + time[1] + "." + file_format
    if name is not None:
        filename = name + filename
    return filename

def visualize_bbox(img, bbox, class_name, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
   
    if class_name == "fallen":
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=FALLEN_COLOR, thickness=thickness)
    else:
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=NO_FALLEN_COLOR, thickness=thickness)
        
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, img_path):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    if img_path is not False:
        plt.savefig(os.path.join("snapshots",img_path))
    
def visualize_images_and_bb(dataset_folder, dataset, rl, save=None):
    for l in rl:
        img_path = dataset['img_path'][l]
        image = cv2.imread(f'{dataset_folder}/{img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        datas = dataset[dataset['img_path'] == img_path]
        bboxes = [[d['x0'], d['y0'], d['x1'], d['y1']] for _, d in datas.iterrows()]
        category_ids = [d['label'] for _,d in datas.iterrows()]
        if save is None:
            visualize(image, bboxes, category_ids, category_id_to_name, False)
        elif save == "virtual_images" or save == "test_images":
            visualize(image, bboxes, category_ids, category_id_to_name, f'{save}/img_{l}.png')
        else:
            raise ValueError(f'save argument can be: None, test_images or virtual_images, not {save}')
        
def visualize_bbox_tensor(img, bbox, class_name):
    if class_name == "fallen":
        color = FALLEN_COLOR
    else:
        color = NO_FALLEN_COLOR
    x0,y0,x1,y1 = bbox

    draw = ImageDraw.Draw(img)
    draw.rectangle(((x0, y0),(x1,y1)), outline=color, width=3)
    draw.text((x0, y0), class_name)
    return img


def visualize_from_tensor(img,  target, category_id_to_name):
    bboxes = target['boxes'].tolist() 
    category_ids = target['labels'].tolist()
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox_tensor(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    
def visualize_from_tensor_and_bb(train_dataset, rl):
    for i in rl:
        img, target = train_dataset[i]
        visualize_from_tensor(img, target, category_id_to_name)
        
# def visualize_prediction(dataset, random_list, model, device, path=None):
#     for l in random_list:
#         img,target = dataset[l]
#         im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
#         model.eval()
#         with torch.no_grad():
#             prediction = model([img.to(device)])
#         boxes = prediction[0]['boxes'].tolist()
#         labels = prediction[0]['labels'].tolist()
#         scores = prediction[0]['scores'].tolist()
#         size = len(target['labels'])

#         if len(labels) > 0:
#             for i in range(size):
#                 if labels[i] == 1:
#                     color = 'green'
#                     text = 'no fall'
#                 elif labels[i] == 2:
#                     color = 'red'
#                     text = 'fall'
#                 x0,y0,x1,y1 = boxes[i]

#                 draw = ImageDraw.Draw(im)
#                 draw.rectangle(((x0, y0),(x1,y1)), outline=color, width=3)
#                 draw.text((x0, y0), text)
#         if path is None:
#             ImageShow.show(im)
#         elif path == "pred_test_images" or path == "pred_virtual_images":
#             #ImageShow.show(im)
#             im.save(f'snapshots/{path}/pred_img_{l}.png')
#         else:
#             raise ValueError(f'save argument can be: None, pred_test_images or pred_virtual_images, not {path}')
            
# ---------------------NEW ACCURACY----------------------------------
def load_saved_model(path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create a Faster R-CNN model without pre-trained
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    num_classes = 3 # wheat or not(background)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained model's head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # load the trained weights
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # move model to the right device
    _ = model.to(device)
    return model, device

def take_prediction(prediction, threshold):
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()

    if len(boxes) == 0:
        return [[0,0,0,0],2,0.]

    if all(l == 1 for l in labels) or all(l == 2 for l in labels):
        if scores[0] < threshold:
            return [[0,0,0,0],2,0.]
        else:
             return [boxes[0],labels[0],scores[0]]
    
    max_no_fall = max((t for t in zip(boxes,labels,scores) if t[1] == 1), key=itemgetter(2))
    max_fall = max((t for t in zip(boxes,labels,scores) if t[1] == 2), key=itemgetter(2))
    max_score = max(max_no_fall, max_fall, key=itemgetter(2))

    if max_score[2] > threshold:
        return max_score
    return [[0,0,0,0],2,0.]

def visualize_prediction(dataset, list_imgs, model, device, path=None, thr=0.7):
    for l in list_imgs:
        img,target = dataset[l]
        with torch.no_grad():
            prediction = model([img.to(device)])
        bb,label,score = take_prediction(prediction[0],thr)
        if label == 1:
            color = "green"
            text = f"no fallen: {score:.3f}"
        else:
            color = "red"
            text = f"fallen: {score:.3f}"
        x0,y0,x1,y1 = bb
        im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        draw = ImageDraw.Draw(im)
        draw.rectangle(((x0, y0),(x1,y1)), outline=color, width=3)
        draw.text((x0, y0), text)
        if path is None:
            ImageShow.show(im)
        else:
            im.save(f'snapshots/{path}{l}.png')
        
def classifier_performance(dataset, model, device, tr = 0.7):
    tn, tp, fn, fp = 0,0,0,0
    for im,target in tqdm(dataset):
        gt_labels = target['labels'].tolist()
        with torch.no_grad():
            prediction = model([im.to(device)])
        _, label, _ = take_prediction(prediction[0], tr)
        if label == 1: 
            if label in gt_labels:
                tn = tn + 1
            else:
                fn = fn + 1
        elif label == 2:
            if label in gt_labels:
                tp = tp + 1        
            else:
                fp = fp + 1
        accuracy = (tp+tn)/(tp+tn+fn+fp)
    return accuracy, tn, tp, fn, fp