from datasets2 import trainset
from validate import validate
import utils
from newmodel2 import NewUNet3
import torch
from torch.utils.data import DataLoader
import numpy as np
np.float_ = np.float32 # importing musdb throws an error if i don't do this ¯\_(ツ)_/¯
import musdb
import os



if __name__=='__main__':
    mus_path = 'E:/musdb18'
    mus_train = musdb.DB(mus_path, subsets='train', split='train')
    mus_valid = musdb.DB(mus_path, subsets='train', split='valid')
    mus_test = musdb.DB(mus_path, subsets='test')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = trainset(mus_train, n_fft=1022, hop_length=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=6, pin_memory=True, pin_memory_device=device)

    model = NewUNet3((2,512,512)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    

    path = 'E:/checkpoint/unet'
    start_epoch, losses = utils.auto_load(path, model, optimizer, overwrite=False)

    num_epochs = 50

    for epoch in range(start_epoch, num_epochs):
        if losses:
            sdr = losses[-1][2].cpu().numpy()
            weights = utils.softmax_weights(sdr, alpha=0.25)
        else:
            weights = [0.25,0.25,0.25,0.25]
        #print('Stem weights: ', weights)

        train_loss, sdr_loss, l1_loss = utils.l1_train(model, optimizer, dataloader, epoch, weights, device)
        utils.save_checkpoint(model, optimizer, epoch+1, (train_loss, None, None, sdr_loss, l1_loss), path)
        valid_loss, valid_sdr = validate(model, mus_test, epoch, sr=22050, n_fft=1022, hop_length=256, chunklen=512)
        utils.save_checkpoint(model, optimizer, epoch+1, (train_loss, valid_loss, valid_sdr, sdr_loss, l1_loss), path)
        losses.append((train_loss, valid_loss, valid_sdr, sdr_loss, l1_loss))
        torch.save(losses, os.path.join(path, 'loss.pth'))
        utils.show_memory_alloc()

