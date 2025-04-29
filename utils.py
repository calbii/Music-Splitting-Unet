import torch
import tqdm
from losses import combo, weighted_l1
import os
import pathlib
import numpy as np



class Averager():
    def __init__(self, sdr=False):
        self.count = 0
        if sdr:
            self.sum = torch.zeros(4, device='cuda')
        else:
            self.sum = 0

    def __len__(self):
        return self.count
    
    def add(self, x):
        self.sum += x
        self.count += 1

    def get(self):
        if self.count > 0:
            return self.sum/self.count
        else:
            return self.sum
        
def train(model, optimizer, loader, epoch, weights, n_fft=2046, hop_length=512, device='cpu'):
    model.train()
    window = torch.hann_window(n_fft, device=device)
    avgloss = Averager()
    avgsdrloss = Averager()
    avgl1loss = Averager()
    pbar = tqdm.tqdm(loader)
    for X, Y, Xphase, ref, mean, std in pbar:
        pbar.set_description(f'Epoch {epoch+1} - Training batch')
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)
        ref, Xphase = ref.to(device, non_blocking=True), Xphase.to(device, non_blocking=True)
        mean, std = mean.to(device, non_blocking=True), std.to(device, non_blocking=True)
        loss, sdr, l1 = combo(ref, pred, Y, Xphase, mean, std, window, n_fft, hop_length, weights)
        loss.backward()
        optimizer.step()
        avgloss.add(loss.item())
        avgsdrloss.add(sdr)
        avgl1loss.add(l1)
        pbar.set_postfix(loss=f'{avgloss.get():.3f}')
        del X,Y,pred,loss,ref,Xphase
        torch.cuda.empty_cache()
    return avgloss.get(), avgsdrloss.get(), avgl1loss.get()

def l1_train(model, optimizer, loader, epoch, weights, device='cpu'):
    model.train()
    avgloss = Averager()
    pbar = tqdm.tqdm(loader)
    for X, Y, Xphase, ref, mean, std in pbar:
        pbar.set_description(f'Epoch {epoch+1} - Training batch')
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_l1(pred, Y, weights)
        loss.backward()
        optimizer.step()
        avgloss.add(loss.item())
        pbar.set_postfix(loss=f'{avgloss.get():.3f}')
        del X, Y, Xphase, ref, mean, std
        torch.cuda.empty_cache()
    return avgloss.get(), None, None

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path='E:/savedtensors/train/checkpoint/timeaugment'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(checkpoint_path, f'{epoch}.pth'))

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.zero_grad()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss'][0]
    print(f'Resuming training from epoch {start_epoch} with average loss {loss:.5f}\n')
    return start_epoch

def show_memory_alloc():
    torch.cuda.synchronize()
    print(f"\nMemory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")

def auto_load(path, model, optimizer, overwrite=False):
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir()
    elif not path.is_dir():
        path.mkdir()
    last_epoch = 0
    epoch_path = ''
    losses = []
    for file in path.iterdir():
        if not file.is_dir():
            if file.suffix == '.pth':
                try:
                    epoch = int(file.stem)
                    if overwrite:
                        file.unlink()
                    elif epoch > last_epoch:
                        last_epoch = epoch
                        epoch_path = file
                except:
                    pass
    if last_epoch > 0:
        load_checkpoint(model, optimizer, epoch_path)
    if path.joinpath('loss.pth').exists():
        if overwrite:
            path.joinpath('loss.pth').unlink()
        else:
            losses = torch.load(path.joinpath('loss.pth'))
    
    return last_epoch, losses

def softmax_weights(sdrs, alpha=0.2, min_weight=0.15):
    sdrs = np.array(sdrs)
    sdrs = 10 * np.log10(sdrs + 1e-8)
    scores = np.exp(-alpha * sdrs)
    weights = scores / np.sum(scores)

    weights = np.maximum(weights, min_weight)
    weights /= np.sum(weights)
    return list(weights)
    
    


