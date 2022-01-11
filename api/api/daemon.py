import os
import sys
import tempfile
import requests
import argparse
from urllib.parse import urlparse
from bottle import route, run, request, debug, HTTPError, HTTPResponse

def init():
    global discriminator_endpoint

    parser = argparse.ArgumentParser(description='A script for testing working of APIs')
    parser.add_argument('--discriminator_endpoint', default=os.environ.get('API_DISCRIMINATOR_ENDPOINT'), type=str, help='discriminator endpoint')
    args = parser.parse_args()

    discriminator_endpoint = args.discriminator_endpoint

@route('/predict', method='POST')
def predict():
    try:
        if not discriminator_endpoint:
            raise Exception('NOT set discriminator endpoint')
    except Exception as e:
        return HTTPError(Exception=e)

    uploads = request.files.getall('upload')

    # if no file is attached
    if uploads is None:
        return HTTPError(status="406 Not Acceptable")

    files = []
    with tempfile.TemporaryDirectory() as dname:
        i = 1
        for upload in uploads:
            temp_file = os.path.join(dname, "temp_file{}".format(i))
            upload.save(temp_file)
            files.append(('upload', ('temp{}.jpg'.format(i), open(temp_file, 'rb'), 'image/jpeg')))
            i += 1

    try:
        response = requests.post(discriminator_endpoint, files=files)
    except Exception as e:
        return HTTPError(exception=e)
    
    if not response.status_code == 200:
        body = response.text
        print(body, file=sys.stderr)
        return HTTPError(status=500, body=body)

    rtn = HTTPResponse(status=200, body=response.content)
    rtn.set_header('Content-Type', 'application/json')
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

if __name__ == "__main__":
    init()
    debug(True)
    run(host='0.0.0.0', port=8080)
