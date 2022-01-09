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
        max_len=100, return_len=False)
    train_loader = data.DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True)

    val_dataset = DisflQA(
        file_name='Datasets/Disfl-QA/dev.json', 
        max_len=100, return_len=False)
    val_loader = data.DataLoader(val_dataset, batch_size=8, num_workers=2)

    src_vocab_emb = WordEmbedding(len(train_dataset.src_vocab), 256)
    tgt_vocab_emb = WordEmbedding(len(train_dataset.tgt_vocab), 256)
    model = LSTM_ED(src_vocab_emb, tgt_vocab_emb, emb_dim=256, hid_dim=256, n_layers=2, dropout=0.2, bidirectional=True).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = CELossShift(ignore_index=3)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = -1
    best_loss = 1e9
    history = {
        'train_loss': [], 
        'val_loss': []
    }
    
    if False:
        start_epoch, stats = load('Checkpoints/lstm_bi_ed.pt', model, optimizer)
        best_loss = stats['val_loss']
        history = history

    n_epochs = 100
    for i in range(start_epoch+1,n_epochs):
        print('Epoch {}:'.format(i))
        train_loss = train(train_loader, model, optimizer, criterion, device='cuda', scaler=scaler, kw_src=['input','output'])
        val_loss = test(val_loader, model, criterion, device='cuda', return_results=False, kw_src=['input','output'])
        
        # Log of loss values
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            save('Checkpoints/lstm_bi_ed.pt', model, optimizer, epoch=i, stats={'val_loss': best_loss, 'history': history})
        
    # --- Test part ---
    test_dataset = DisflQA(
        file_name='Datasets/Disfl-QA/test.json', 
        max_len=100, return_len=False, infer=True)
    test_loader = data.DataLoader(test_dataset, batch_size=8, num_workers=2)

    
    _, outputs, targets = test(test_loader,model,device='cuda',return_results=True)
    train_dataset.tgt_vocab.decode(outputs[11].numpy().tolist())
    train_dataset.tgt_vocab.decode(targets[11].numpy().tolist())
