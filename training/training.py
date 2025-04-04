from tqdm import tqdm

def train_single_epoch(dataloader, model, loss_fct, optimizer, device, logger):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    running_loss = 0.0
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fct(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch % 6000 == 5999:    
            logger.info(f'[{batch + 1:5d}/{len(dataloader):>5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0