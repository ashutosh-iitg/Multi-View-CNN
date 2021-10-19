"""Evaluates the model"""
import numpy as np
from tqdm import tqdm

import wandb
import torch
import utils
from utils import calculate_accuracy

def evaluate(dataloader, model, criterion, epoch, params):
    metric_monitor = utils.MetricMonitor()
    model.eval()

    stream = tqdm(dataloader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params.device, non_blocking=True)
        target = target.to(params.device, non_blocking=True)

        # compute model output and loss
        with torch.no_grad():
            output = model(images)

        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)

        stream.set_description(
            "Epoch: {epoch}/{epochs}. Validation. {metric_monitor}".format(epoch=epoch, epochs=params.epochs, metric_monitor=metric_monitor)
        )
        wandb.log({"epoch":epoch, "val_accuracy": accuracy, "val_loss": loss.item()})
    
    '''if (epoch%5==0):
        torch.onnx.export(model, images, "model.onnx")
        wandb.save("model.onnx")'''

    return metric_monitor()