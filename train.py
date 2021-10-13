"""Train the model"""
import numpy as np
import torch
from tqdm import tqdm

import utils
from utils import calculate_accuracy

def train(dataloader, model, criterion, optimizer, epoch, params):
    metric_monitor = utils.MetricMonitor()
    model.train()

    stream = tqdm(dataloader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params.device, non_blocking=True)
        target = target.to(params.device, non_blocking=True)

        output = model(images)

        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stream.set_description(
            "Epoch: {epoch}/{epochs}. Train.      {metric_monitor}".format(epoch=epoch, epochs=params.epochs, metric_monitor=metric_monitor)
        )

    return metric_monitor()