import torch
from tqdm import tqdm
import config

import numpy as np


def intersection_over_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum() / union.sum()


def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    train_loader = tqdm(train_loader, desc="batches")
    for it in train_loader:
        data = it["chip"].type(torch.FloatTensor)
        targets = it["label"]
        data = data.to(config.device)
        targets = targets.type(torch.LongTensor).to(config.device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss = loss.to(config.device)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loader.set_postfix(loss=loss.item())


def val_fn(val_loader, model):
    iou_list = []
    val_loader = tqdm(val_loader)
    with torch.no_grad():
        model.eval()
        for it in val_loader:
            input_image = it["chip"].type(torch.FloatTensor).to(config.device)
            true_mask = it["label"].squeeze()
            predicted_mask = model(input_image)
            predicted_mask = torch.argmax(predicted_mask, dim=1).squeeze()
            batch_iou = intersection_over_union(predicted_mask.detach().to("cpu"), true_mask)
            iou_list.append(batch_iou)
            val_loader.set_postfix(iou=sum(iou_list) / len(iou_list))
    model.train()
    return sum(iou_list) / len(iou_list)
