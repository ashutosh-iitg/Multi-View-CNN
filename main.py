import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import utils
from data import PreProcessing, PlantDataset
from model import MVCNN
from train import train
from evaluate import evaluate

cudnn.benchmark = True

def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="Training settings and parameters")
    parser.add_argument("--csv-dir", type=str, default="data.csv", help="Path to csv file")
    parser.add_argument("--image-dir", type=str, default="images", help="Full path to image directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Full path to output directory")
    parser.add_argument("--params-path", type=str, default="config/hparams.json", help="Path to hyperparameters json file")
    parser.add_argument('--model', type=str, default='resnet34', help='Model architecture to be used for training')
    parser.add_argument('--encoding', type=str, default='utf-8', help='CSV file encoding')
    parser.add_argument('--debug', nargs='?', const=True, default=False, help='run script in debug mode')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--resume-best', nargs='?', const=True, default=False, help='resume training from best saved model')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def get_transform(split="train"):
    if split=="train":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3),
                A.OpticalDistortion (distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, p=0.3),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
                ToTensorV2(p=1.0),
            ]
        )
    else:
        transform = A.Compose(
            [
                ToTensorV2(p=1.0),
            ]
        )
    return transform

def main(opt):
    # Load the parameters from json file
    json_path = opt.params_path
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    if params.cuda:
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')
        
    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # set logger
    utils.set_logger(os.path.join(opt.output_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    data = PreProcessing(csv_dir=opt.csv_dir)
    train_df, test_df, label_dict = data.train_df, data.test_df, data.label_dct

    # update params
    params.mode = opt.model
    params.num_targets = len(label_dict)

    # create dataset and dataloader
    train_dataset = PlantDataset(train_df, opt.image_dir, params, transform=get_transform(split="train"))
    #valid_dataset = BagDataset(X_valid, y_valid, opt.image_dir, transform=get_transform(split="valid")) 
    test_dataset = PlantDataset(test_df, opt.image_dir, params, transform=get_transform(split="test"))

    train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    #valid_dl = DataLoader(valid_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False, pin_memory=True)
    
    # data-loading completed
    logging.info("- done.")

    # Define the model
    model = MVCNN(num_classes=params.num_targets, pretrained=True)
    model = model.to(params.device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # fetch loss function 
    criterion = nn.CrossEntropyLoss().to(params.device)

    # reload weights from restore_file if specified
    if opt.resume_best:
        restore_path = os.path.join(opt.output_dir, 'best.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    elif opt.resume:
        restore_path = os.path.join(opt.output_dir, 'last.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.epochs))
    best_acc = 0
    for epoch in range(1, params.epochs+1):
        # one full pass over the training set
        train_metrics = train(train_dl, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, params=params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(test_dl, model=model, criterion=criterion, epoch=epoch, params=params)
        scheduler.step(val_metrics["Loss"])

        is_best = val_metrics["Accuracy"] >= best_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=opt.output_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_acc = val_metrics["Accuracy"]

        # log training and validation metrics
        logging.info("- Train metrics:      {}/{}".format(epoch, params.epochs) + " | ".join("{}:{:05.3f}".format(k, v) for k, v in train_metrics.items()))
        logging.info("- Validation metrics: {}/{}".format(epoch, params.epochs) + " | ".join("{}:{:05.3f}".format(k, v) for k, v in val_metrics.items()))

        # test the performance of model on test set after every 10 epochs
        '''if (epoch % 10 == 0):
            logging.info("Performance on test set after epoch {}".format(epoch))
            test_metrics = evaluate(test_dl, model=model, criterion=criterion, epoch=epoch, params=params)
            logging.info("- Test metrics: {}/{}".format(epoch, params.epochs) + " | ".join("{}:{:05.3f}".format(k, v) for k, v in test_metrics.items()))
        '''
    logging.info("Training completed ....")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)