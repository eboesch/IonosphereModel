import torch
from torch.utils.data import DataLoader
from torch.nn import Module

def test(dataloader: DataLoader, model: Module, loss_fct: Module, device: torch.device) -> float:
    """
    Evaluates the given model on the data in the dataloader using the given loss function.

    Returns the average evaluation loss.
    """
    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fct(pred, y).item()

    test_loss /= num_batches

    return test_loss
