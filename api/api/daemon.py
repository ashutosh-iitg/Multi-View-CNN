import os
import sys
import requests
import argparse
import tempfile

from urllib.parse import urlparse
from bottle import route, run, request, debug, HTTPError, HTTPResponse


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

    upload1 = request.files.get('upload1')
    upload2 = request.files.get('upload2')

    # no file is attached
    if upload1 is None or upload2 is None:
        return HTTPError(status="406 Not Acceptable")

    with tempfile.TemporaryDirectory() as dname:
        temp_file1 = os.path.join(dname, "temp_file1")
        temp_file2 = os.path.join(dname, "temp_file2")
        upload1.save(temp_file1)
        upload2.save(temp_file2)
    file_names = [temp_file1,temp_file2]
    files = [('file', open(f, 'rb')) for f in file_names]
    
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