# Multi-View-CNN
A CNN network designed to take benefit of complementary features provided by multiple views of an object

## How to run

### start a daemon in Docker
```
$ docker-compose build
$ docker-compose up -d
```

### Input two images to API via HTTP
```
$ curl -X POST http://localhost:8080/predict -F "upload=@99999.jpg;type=image/jpeg" -F "upload=@99998.jpg;type=image/jpeg"
```
