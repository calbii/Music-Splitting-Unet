import numpy as np
np.float_ = np.float32
import musdb
import torch
import os
import librosa
import fast_bss_eval
from torchmetrics.functional.audio.sdr import source_aggregated_signal_distortion_ratio as sa_sdr
import tqdm
import torchaudio

def getrefs(track, sr):
    ref = torch.stack([torch.tensor(librosa.resample(track.targets[stem].audio.T, orig_sr=track.rate, target_sr=sr, fix=True), dtype=torch.float32) for stem in ['drums','bass','other','vocals']], dim=0)
    return ref

def getstft(track, sr, n_fft, hop_length, window):
    mix = librosa.resample(track.audio.T, orig_sr=track.rate, target_sr=sr, fix=True)
    X = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length, window=window)
    return X

def padX(X, chunklen):
    pad = chunklen - (X.shape[-1] % chunklen)
    X = np.pad(X, ((0,0), (0,0), (pad,0)), mode='constant')
    return X, pad

def getchunk(X, index, chunklen):
    start = index * chunklen
    end = start + chunklen
    return X[...,start:end]

def evalchunk(model, X):
    pred = model(X.unsqueeze(0))
    return pred.squeeze(0)

def stitchpreds(preds):
    fullpred = torch.cat(preds, dim=-1)
    return fullpred

def normalize(X, device='cuda'):
    X = torch.log1p(torch.abs(torch.as_tensor(X, device=device)))
    mean = X.mean(dim=(0,2), keepdim=True)
    std = X.std(dim=(0,2), keepdim=True)
    X = (X - mean) / (std + 1e-8)
    return X, mean, std

def denormalize(pred, mean, std):
    pred = pred * std.unsqueeze(0) + mean.unsqueeze(0)
    return torch.expm1(pred)

def predictsong(model, track, device, sr, n_fft, hop_length, chunklen):
    X = getstft(track=track, sr=sr, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).cpu().numpy())
    Xphase = torch.tensor(np.angle(X), device=device)
    X, pad = padX(X, chunklen)
    if X.shape[-1] % chunklen != 0:
        raise ValueError('Total length not divisible by chunk length')
    preds = []
    for i in range(X.shape[-1]//chunklen):
        Xchunk = getchunk(X, i, chunklen)
        Xchunk, mean, std = normalize(Xchunk)
        pred = evalchunk(model, Xchunk)
        pred = denormalize(pred, mean, std)
        preds.append(pred)
    preds = stitchpreds(preds)[...,pad:]
    preds = preds * torch.exp(1j * Xphase.unsqueeze(0))
    preds = torch.stack([torch.istft(preds[i,...], n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=device)) for i in range(preds.shape[0])], dim=0)
    ref = getrefs(track, sr).to(device)
    loss = -torch.mean(sa_sdr(torch.permute(preds, (1,0,2)), torch.permute(ref, (1,0,2))))
    sdrs = fast_bss_eval.si_sdr(ref, preds)
    return loss, sdrs

def writeout(model, track, device, sr, n_fft, hop_length, chunklen, path):
    X = getstft(track=track, sr=sr, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).cpu().numpy())
    Xphase = torch.tensor(np.angle(X), device=device)
    X, pad = padX(X, chunklen)
    if X.shape[-1] % chunklen != 0:
        raise ValueError('Total length not divisible by chunk length')
    preds = []
    for i in range(X.shape[-1]//chunklen):
        Xchunk = getchunk(X, i, chunklen)
        Xchunk, mean, std = normalize(Xchunk)
        pred = evalchunk(model, Xchunk)
        pred = denormalize(pred, mean, std)
        preds.append(pred)
    preds = stitchpreds(preds)[...,pad:]
    preds = preds * torch.exp(1j * Xphase.unsqueeze(0))
    preds = torch.stack([torch.istft(preds[i,...], n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=device)) for i in range(preds.shape[0])], dim=0)
    preds = preds.cpu()

    os.makedirs(path, exist_ok=True)
    torchaudio.save(os.path.join(path, 'drums.wav'), preds[0,...], sample_rate=22050)
    torchaudio.save(os.path.join(path, 'bass.wav'), preds[1,...], sample_rate=22050)
    torchaudio.save(os.path.join(path, 'other.wav'), preds[2,...], sample_rate=22050)
    torchaudio.save(os.path.join(path, 'vocals.wav'), preds[3,...], sample_rate=22050)


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

def validate(model, db, epoch, sr=22050, n_fft=1022, hop_length=512, chunklen=512):
    avgloss = Averager()
    avgsdrs = Averager(sdr=True)
    pbar = tqdm.tqdm(db)
    with torch.no_grad():
        model.eval()
        for track in pbar:
            pbar.set_description(f'Epoch {epoch+1} - Validating')
            loss, sdrs = predictsong(model, track, 'cuda', sr, n_fft, hop_length, chunklen)
            sdrs = 10 ** (sdrs/10)
            sdrs = torch.mean(sdrs, dim=1)
            avgloss.add(loss.item())
            avgsdrs.add(sdrs)
            sdr = 10*torch.log10(avgsdrs.get()+1e-8)
            pbar.set_postfix(loss=f'{avgloss.get():.3f}',
                             drums=f'{sdr[0]:.3f}',
                             bass=f'{sdr[1]:.3f}',
                             other=f'{sdr[2]:.3f}',
                             vocals=f'{sdr[3]:.3f}')
            del loss, sdrs, sdr
            torch.cuda.empty_cache()
        return avgloss.get(), avgsdrs.get()         


if __name__=="__main__":
    pass




