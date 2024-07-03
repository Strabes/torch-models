import time
import torch
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler=None):
    """
    Training a basic torch text model

    Parameters
    ----------
    model: torch.nn.Module

    train_dataloader: torch.utils.data.DataLoader
    """
    t0_epoch = time.time()
    model.train()

    epoch_loss = 0
    
    for batch in tqdm(train_dataloader):
        numeric = batch.numeric.to(device)
        categorical = batch.categorical.to(device)
        text = [text.to(device) for text in batch.text]
        targets = batch.target.to(device)
        loss_weights = batch.loss_weights.to(device)

        optimizer.zero_grad()

        preds = model(numeric, categorical, text)
        loss = loss_fn(preds, targets, loss_weights)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    mean_train_loss = epoch_loss / len(train_dataloader)

    time_elapsed = time.time() - t0_epoch

    return mean_train_loss, time_elapsed

def evaluation(model, dataloader, loss_fn, device):
    t0 = time.time()
    model.eval()

    eval_loss = []

    for batch in tqdm(dataloader):
        numeric = batch.numeric.to(device)
        categorical = batch.categorical.to(device)
        text = [text.to(device) for text in batch.text]
        targets = batch.target.to(device)
        loss_weights = batch.loss_weights.to(device)

        with torch.no_grad():
            preds = model(numeric, categorical, text)

        # Compute loss
        loss = loss_fn(preds, targets, loss_weights)
        eval_loss.append(loss.item())

    eval_loss = np.mean(eval_loss)

    time_elapsed = time.time() - t0

    return eval_loss, time_elapsed