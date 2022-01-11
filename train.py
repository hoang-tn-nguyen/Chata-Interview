import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from datasets import DisflQA
from models import MachineTranslationModel, LSTM_ED, WordEmbedding
from utils import train, test, save, load
from losses import CELossShift

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)

MODEL_NAME = 'PERFORMER_FINETUNE'
TRAIN_PHASE = False
RELOAD_MODEL = True
FINETUNE_MODEL = False
EPOCHS = 100

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
    
    if MODEL_NAME in ['LSTM_ED', 'LSTM_ED_FINETUNE']:
        model = LSTM_ED(src_vocab_emb, tgt_vocab_emb, emb_dim=256, hid_dim=256, n_layers=4, dropout=0.1).cuda()
    elif MODEL_NAME in ['LSTM_BI_ED', 'LSTM_BI_ED_FINETUNE']:
        model = LSTM_ED(src_vocab_emb, tgt_vocab_emb, emb_dim=256, hid_dim=256, n_layers=4, dropout=0.1, bidirectional=True).cuda()
    elif MODEL_NAME in ['PERFORMER', 'PERFORMER_FINETUNE']:
        model = MachineTranslationModel(src_vocab_emb, tgt_vocab_emb, latent_embed=100, embed_dim=256, ffwd_dim=256, num_heads=1, num_enc_layers=6, num_dec_layers=6, dropout=0.1).cuda()
    else:
        raise ValueError('Unknown MODEL_NAME')

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = CELossShift(ignore_index=3)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = None

    start_epoch = -1
    best_loss = 1e9
    history = {
        'train_loss': [], 
        'val_loss': []
    }

    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    if RELOAD_MODEL:
        start_epoch, stats = load('Checkpoints/{}.pt'.format(MODEL_NAME), model, optimizer)
        best_loss = stats['val_loss']
        history = stats['history']
        
        # Lazy update the graph
        plt.ylabel('Loss Value')
        plt.xlabel('Number of Epoch') 
        plt.plot(np.arange(len(history['train_loss'])), history['train_loss'], linestyle='--', color='g', label='Train Loss')
        plt.plot(np.arange(len(history['val_loss'])), history['val_loss'], linestyle='--', color='r', label='Validation Loss')
        plt.legend() 
        plt.savefig('Results/Loss_{}.png'.format(MODEL_NAME))
        plt.show()

    if FINETUNE_MODEL:
        # Reinitialize the optimizer, treat this as fine-tuning a pretrained model
        optimizer = optim.AdamW(model.parameters(), lr=3e-4) # Reinitialize the optimizer
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20], gamma=0.1) # set up lr_scheduler
        MODEL_NAME += '_FINETUNE' # avoid overwriting the original model
        EPOCHS += 50 # just in case the model needs more epochs to run

    if TRAIN_PHASE:
        for i in range(start_epoch+1,EPOCHS):
            print('Epoch {}:'.format(i))
            train_loss = train(train_loader, model, optimizer, criterion, scheduler, device='cuda', scaler=scaler, kw_src=['input','output'])
            val_loss = test(val_loader, model, criterion, device='cuda', return_results=False, kw_src=['input','output'])
            
            # Log of loss values
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                save('Checkpoints/{}.pt'.format(MODEL_NAME), model, optimizer, epoch=i, stats={'val_loss': best_loss, 'history': history})
        
    else:
        test_dataset = DisflQA(
            file_name='Datasets/Disfl-QA/test.json', 
            max_len=100, return_len=False, infer=True)
        test_loader = data.DataLoader(test_dataset, batch_size=128, num_workers=2)

        _, outputs, targets = test(test_loader,model,device='cuda',return_results=True)
        outputs = outputs.numpy()
        targets = targets.numpy()

        log_input = open('Results/{}_log_inputs.txt'.format(MODEL_NAME), 'w', encoding='utf-8')
        log_output = open('Results/{}_log_outputs.txt'.format(MODEL_NAME), 'w', encoding='utf-8')
        log_target = open('Results/{}_log_targets.txt'.format(MODEL_NAME), 'w', encoding='utf-8')

        for i in range(len(test_dataset)):
            str_input = test_dataset.src_vocab.decode(test_dataset[i][0].tolist())
            str_target = test_dataset.tgt_vocab.decode(test_dataset[i][1].tolist())

            post_process_output = []
            for j in range(len(outputs[i])):
                post_process_output.append(outputs[i][j])
                if outputs[i][j] == 2:
                    break
            post_process_output = np.array(post_process_output)        
            str_output = test_dataset.tgt_vocab.decode(post_process_output.tolist())
        
            log_input.write(str_input + '\n')
            log_output.write(str_output + '\n')
            log_target.write(str_target + '\n')

        log_input.close()
        log_output.close()
        log_target.close()

