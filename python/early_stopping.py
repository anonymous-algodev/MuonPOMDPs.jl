import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
