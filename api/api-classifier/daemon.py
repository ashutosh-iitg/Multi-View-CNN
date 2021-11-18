import os
from urllib import parse
import cv2
import sys
import json
import tempfile
import argparse

from torch._C import default_generator
import predictor
from bottle import route, run, request, debug, HTTPError, HTTPResponse

import torch
import numpy as np
from model import MVCNN
from albumentations.pytorch.transforms import ToTensorV2


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
        return HTTPError(body=e, exception=ValueError)

@route('/discriminate', method='POST')
def discriminate():
    try:
        uploads= request.files.getall('upload')

        if not uploads:
            return HTTPError(status='406 Not Acceptable')

        # how to post multiple files
        images = []
        with tempfile.TemporaryDirectory() as dname:
            i = 0
            for upload in uploads:
                temp_file = os.path.join(dname, "temp_file")
                upload.save(temp_file)
                with open(temp_file, "rb") as f:
                    image = np.asarray(bytearray(f.read()), dtype="uint8")
                images.append(get_image(image))
        images = torch.stack(images, dim=0)

        pred = predict(images)
        content = {
            'Genus': pred
        }
        show_json = json.dumps(content, ascii=False)
        rtn = HTTPResponse(status=200, body=show_json)
        rtn.set_header('Content-Type', 'application/json')
        return rtn
    except ValueError as e:
        print(e, file=sys.stderr)
        return HTTPError(body=e, exception=ValueError)

@route('/health_check', method='GET')
def health_check():
    return "I'm OK."

if __name__=='__main__':
    init()
    # debug(True)
    run(host='0.0.0.0', port=8091)