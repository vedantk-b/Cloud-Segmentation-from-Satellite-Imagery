from data import CloudDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
import config
import utils

train_x = pd.read_csv("data/train_x.csv").drop(columns="Unnamed: 0")
train_y = pd.read_csv("data/train_y.csv").drop(columns="Unnamed: 0")
val_x = pd.read_csv("data/val_x.csv").drop(columns="Unnamed: 0")
val_y = pd.read_csv("data/val_y.csv").drop(columns="Unnamed: 0")
training_dataset = CloudDataset(train_x, config.bands, train_y,
                                config.train_transforms)
validation_dataset = CloudDataset(val_x, config.bands, val_y,
                                  config.val_transforms)
train_loader = DataLoader(dataset=training_dataset, batch_size=config.train_batch_size,
                          shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(dataset=validation_dataset, batch_size=config.val_batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
model = config.model
model.to(config.device)
loss_fn = config.loss_function
optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
scaler = config.scaler
best_iou = -1
for i in range(config.epochs):
    utils.train_fn(train_loader, model, optimizer, loss_fn, scaler)
    iou = utils.val_fn(val_loader, model)
    if iou > best_iou:
        torch.save(model.state_dict(), f"model.pth")
        torch.save(optimizer.state_dict(), f"optim.pth")
        best_iou = iou

print("-"*50)
print(f"Training done the best performing model has an mIoU score of : {best_iou}")
