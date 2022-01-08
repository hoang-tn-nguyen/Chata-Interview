import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from datasets import DisflQA
from models import LSTM_ED, WordEmbedding
from utils import train, test, save, load
from losses import CELossShift

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)

if __name__ == '__main__':
    train_dataset = DisflQA(
        file_name='Datasets/Disfl-QA/train.json', 
        vocab_file='Datasets/Disfl-QA/spm.model', 
        max_len=100, return_len=False)
    train_loader = data.DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True)

    val_dataset = DisflQA(
        file_name='Datasets/Disfl-QA/dev.json', 
        vocab_file='Datasets/Disfl-QA/spm.model', 
        max_len=100, return_len=False)
    val_loader = data.DataLoader(val_dataset, batch_size=8, num_workers=2)

    vocab_emb = WordEmbedding(len(train_dataset.vocab), 512)
    model = LSTM_ED(vocab_emb, emb_dim=512, hid_dim=512, n_layers=4, dropout=0.2).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = CELossShift(ignore_index=3)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = 1e9
    history = {
        'train_loss': [], 
        'val_loss': []
    }
    
    n_epochs = 100
    for i in range(n_epochs):
        print('Epoch {}:'.format(i))
        train_loss = train(train_loader, model, optimizer, criterion, device='cuda', scaler=scaler, kw_src=['input','output'])
        val_loss = test(val_loader, model, criterion, device='cuda', return_results=False, kw_src=['input','output'])
        
        # Log of loss values
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            save('Checkpoints/lstm_ed.pt', model, optimizer, epoch=i, stats={'val_loss': best_loss, 'history': history})
        
    # --- Test part ---
    # test_dataset = DisflQA(
    #     file_name='Datasets/Disfl-QA/test.json', 
    #     vocab_file='Datasets/Disfl-QA/spm.model', 
    #     max_len=100, return_len=False, infer=True)
    # test_loader = data.DataLoader(test_dataset, batch_size=8, num_workers=2)

    # load('Checkpoints/lstm_ed.pt', model)
    # _, outputs, targets = test(test_loader,model,device='cuda',return_results=True)
    # train_dataset.vocab.decode(outputs[3].numpy().tolist())
    # train_dataset.vocab.decode(targets[3].numpy().tolist())
