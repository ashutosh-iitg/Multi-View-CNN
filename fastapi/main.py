import os
import cv2
import sys
import json
import argparse

from typing import ItemsView, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

import uvicorn
import predictor

import torch
import numpy as np
from model import MVCNN
from albumentations.pytorch.transforms import ToTensorV2

app = FastAPI()

def init():
    global predict_model
    global label_dict

    parser = argparse.ArgumentParser(description='A script for prediction using MVCNN model')
    parser.add_argument('--model-path', default='output/best.pth.tar', type=str, help='prediction model file path.')
    parser.add_argument('--label-path', default='output/label_dict.json', type=str, help='label json file path.')

    args = parser.parse_args()

    with open(args.label_path, 'r') as f:
        label_dict = json.load(f)
    
    predict_model = MVCNN(num_classes=len(label_dict), pretrained=False)
    predictor.load_model(args.model_path, predict_model)

def get_image(image_path):
    image = cv2.imdecode(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (300,300))
    image /= 255.0

    transform = ToTensorV2(p=1.0)
    image = transform(image=image)['image']
    image = image.unsqueeze(0)
    
    return image

def predict(image):
    try:
        inverse_dict = {v: k for k,v in label_dict.items()}
        pred = predictor.predict(predict_model, image)[0]
        pred = inverse_dict[pred]
        print(pred)
        return pred
    except ValueError as e:
        print(e, file=sys.stderr)
        return HTTPException(body=e, exception=ValueError)

@app.post("/predict/")
async def bag_predict(files: List[UploadFile] = File(...)):
    images = [get_image(np.asarray(bytearray(await file.read()))) for file in files]
    images = torch.stack(images, dim=0)
    pred = predict(images)
    return {"Genus": pred}

if __name__=='__main__':
    init()
    # debug(True)
    uvicorn.run(app, host="0.0.0.0", port=8080)