import os
import sys
import json
import requests
import argparse
import tempfile

from urllib.parse import urlparse
from bottle import route, run, request, debug, HTTPError, HTTPResponse
from albumentations.pytorch.transforms import ToTensorV2


def init():
    global discriminator_endpoint

    parser = argparse.ArgumentParser(description="A script for prediction using MVCNN model")
    parser.add_argument('--discriminator-endpoint', default=os.environ.get('API_DISCRIMINATOR_ENDPOINT'), type=str, help='discrimimnator endpoint')
    args = parser.parse_args()

    discriminator_endpoint = args.discriminator_endpoint

@route('/predict', method='POST')
def predict():
    try:
        if not discriminator_endpoint:
            raise Exception('NOT set discriminator endpoint')
    except Exception as e:
        return HTTPError(Exception=e)

    # upload = request.files.get('upload')
    # Reference: https://stackoverflow.com/questions/31642717/python-bottle-multiple-file-upload/31644337#comment51232480_31642717
    uploads = request.files.getall('upload')

    # Check if file is attached
    if uploads is None:
        return HTTPError(status='406 Not Accetable')

    # how to post multiple files
    files = []
    with tempfile.TemporaryDirectory() as dname:
        i = 0
        for upload in uploads:
            temp_file = os.path.join(dname, "temp_file")
            upload.save(temp_file)
            with open(temp_file, "rb") as f:
                image = f.read()
                files.append(('upload', ('temp{}.jpg'.format(i), image, 'image/jpeg')))
            i += 1
    
    # call detect API
    try:
        response = requests.post(discriminator_endpoint, files=files)
    except Exception as e:
        return HTTPError(Exception=e)

    if not response.status_code==200:
        body = response.text
        print(body, file=sys.stderr)
        return HTTPError(status=500, body=body)

    rtn = HTTPResponse(status=200, body=response.content)
    rtn.set_header('Content-type', 'application/json')
    return rtn

@route('/health_check', method='GET')
def health_check():
    try:
        if not discriminator_endpoint:
            raise Exception('NOT set discriminator endpoint')

        # check discriminator
        rst = urlparse(discriminator_endpoint)
        discriminator_health_check_endpoint = rst.scheme + '://' + rst.netloc + '/health_check'

        response = requests.get(discriminator_health_check_endpoint)

        if not response.status_code == 200:
            body = response.text
            print(body, file=sys.stderr)
            return HTTPError(status=500, body=body)

    except Exception as e:
        return HTTPError(Exception=e)

    return "I'm OK."

if __name__=="_main__":
    init()
    debug(True)
    run(host='0.0.0.0', port=8080)