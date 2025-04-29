from torch.utils.data import Dataset
import numpy as np
np.float_ = np.float32
import musdb
import torch
import random
import librosa

def getaud(track, sr):
    mix = librosa.resample(track.audio.T, orig_sr=track.rate, target_sr=sr)
    stems = {
        'drums': librosa.resample(track.targets['drums'].audio.T, orig_sr=track.rate, target_sr=sr, fix=True),
        'bass': librosa.resample(track.targets['bass'].audio.T, orig_sr=track.rate, target_sr=sr, fix=True),
        'other': librosa.resample(track.targets['other'].audio.T, orig_sr=track.rate, target_sr=sr, fix=True),
        'vocals': librosa.resample(track.targets['vocals'].audio.T, orig_sr=track.rate, target_sr=sr, fix=True),
    }
    return mix, stems

def stft(mix, stems, n_fft, hop_length, window):
    X = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length, window=window)
    Y = np.stack([librosa.stft(stems[stem], n_fft=n_fft, hop_length=hop_length, window=window) for stem in list(stems.keys())], axis=0)
    return X, Y

def trimends(X, Y, size: tuple):
    return X[...,size[0]:-size[1]], Y[...,size[0]:-size[1]]

def prepoutputs(X, Y, ref):
    Xphase = torch.tensor(X).angle()
    X = torch.log1p(torch.tensor(X).abs())
    Y = torch.log1p(torch.tensor(Y).abs())
    mean = X.mean(dim=(0,2), keepdim=True)
    std = X.std(dim=(0,2), keepdim=True)
    X = (X - mean) / (std + 1e-8)
    Y = (Y - mean.unsqueeze(0)) / (std.unsqueeze(0) + 1e-8)
    ref = torch.stack([torch.tensor(ref[stem]) for stem in ref.keys()], dim=0)
    return X, Y, Xphase, ref, mean, std
    

class trainset(Dataset):
    def __init__(self, db, sr=22050, n_fft=2046, hop_length=512):
        self.db = db
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunklen = np.floor((511*hop_length) * (44100/sr))
        self.window = torch.hann_window(n_fft).cpu().numpy()
        self.tracksizes = [(track.samples // self.chunklen) for track in self.db]
        self.num_chunks = sum(self.tracksizes)
        self.eps = 1e-8

    def __len__(self):
        return int(self.num_chunks)
    
    def __getitem__(self, index):
        song_idx = 0
        while index >= self.tracksizes[song_idx]:
            index -= self.tracksizes[song_idx]
            song_idx += 1

        track = self.db[song_idx]
        tot_dur = track.samples/track.rate
        chunk_dur = self.chunklen/track.rate
        chunk_start = random.uniform(0, tot_dur - chunk_dur)
        track.chunk_duration = chunk_dur
        track.chunk_start = chunk_start
        
        track2 = self.db[random.randint(0, len(self.db)-1)]
        tot_dur = track2.samples/track2.rate
        track2.chunk_duration = chunk_dur
        track2.chunk_start = random.uniform(0, tot_dur - chunk_dur)

        mix1, stems1 = getaud(track, self.sr)
        mix2, stems2 = getaud(track2, self.sr)

        mix = np.clip(mix1 + mix2, a_min=-1.0, a_max=1.0)
        stems = {key: np.clip(stems1[key] + stems2[key], a_min=-1.0, a_max=1.0) for key in stems1}

        X, Y = stft(mix, stems, self.n_fft, self.hop_length, self.window)

        return prepoutputs(X, Y, stems)
    




        



        
