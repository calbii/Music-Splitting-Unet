import torch
import fast_bss_eval
import torch.nn.functional as F
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio as sa_sdr


def sdrs(ref, pred, xphase, window): # unused
    pred = torch.exp(pred) * torch.exp(1j * xphase.unsqueeze(1))
    est = torch.stack([torch.istft(pred[0,stem,...], n_fft=1022, hop_length=512, window=window) for stem in range(4)], dim=0)
    sdr = 10 ** (fast_bss_eval.si_sdr(ref.squeeze(), est)/10)
    return torch.mean(sdr, dim=1)

def spectralconvergence(pred, Y): # unused
    return torch.linalg.matrix_norm(pred-Y, ord='fro', dim=(-2,-1))/torch.linalg.matrix_norm(Y, ord='fro', dim=(-2,-1))

def sa_sdr_loss(ref, pred, window, n_fft=2046, hop_length=512, weights=[1.0,1.0,1.0,1.0]):
    est = torch.empty_like(ref) # (batch, stems, channels, samples)
    for i in range(est.shape[0]):
        for j in range(est.shape[1]):
            est[i,j,...] = torch.istft(pred[i,j,...], n_fft=n_fft, hop_length=hop_length, window=window) * weights[j]
    est = torch.permute(est, (0,2,1,3)) # permute so norms are aggregated across stems instead of channels
    ref = torch.permute(ref, (0,2,1,3))
    sdr = sa_sdr(est, ref) # (batch, channels)
    loss = -torch.mean(sdr)
    return loss

def weighted_l1(pred, Y, weights = [1.0,1.0,1.0,1.0]): # (batch, stems, channels, bins, frames)
    loss = F.l1_loss(pred, Y, reduction='none')
    loss = loss.mean(dim=(0, 2, 3, 4))
    weights = torch.as_tensor(weights, device = loss.device)
    loss = (loss * weights).sum() / weights.sum()
    return loss.mean()

def combo(ref, pred, Y, xphase, mean, std, window, n_fft=2046, hop_length=512, weights=[1.0,1.0,1.0,1.0], alpha=0.5):
    l1 = weighted_l1(pred, Y, weights)
    pred = pred * std.unsqueeze(1) + mean.unsqueeze(1)
    pred = torch.expm1(pred) * torch.exp(1j * xphase.unsqueeze(1))
    sdrloss = sa_sdr_loss(ref, pred, window, n_fft, hop_length) # including weights seemed to negatively impact performance
    #l1 = F.l1_loss(pred, Y)
    return alpha * sdrloss + (1-alpha) * l1, sdrloss.item(), l1.item()