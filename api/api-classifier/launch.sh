#!/bin/sh

# Sets default values for the env varailbles
LEARNING_LABEL='https://auto-assessing.s3-ap-northeast-1.amazonaws.com/resnet/resnet_labels_platform.json'
LEARNING_MODEL='https://auto-assessing.s3-ap-northeast-1.amazonaws.com/resnet/resnet_models_platform.h5'

# Downloads model weights and config files.
mkdir output/

curl -L $LEARNING_LABEL -o output/label_dict.json
curl -L $LEARNING_MODEL -o output/best.pth.tar

# for local test
# mv models.h5 weights/bcnn_model.h5
# mv labels.json configs/watch_label.json

# Starts supervisord
exec /usr/bin/supervisord