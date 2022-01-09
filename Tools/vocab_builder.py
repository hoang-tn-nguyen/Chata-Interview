import json
import sentencepiece as spm

json_file = open('../Datasets/Disfl-QA/train.json')
data = json.load(json_file)

original_file = open('original.txt','w',encoding='utf-8')
disfluent_file = open('disfluent.txt','w',encoding='utf-8')

# Extract tokens & frequency from sentences
for k,v in data.items():

    original_file.write(v['original'].lower() + '\n')
    disfluent_file.write(v['disfluent'].lower() + '\n')

original_file.close()
disfluent_file.close()

# Build vocabulary with sentencepiece
import glob
punc_list = ['`','~','!','@','#','$','%','^','&','*','-','_','+','=',
             '\\','|',':',';','"','\'',',','.','?','/',
             '(',')','{','}','[',']','<','>'] # punctuation

spm.SentencePieceTrainer.Train(
    input='disfluent.txt', 
    model_prefix='../Datasets/Disfl-QA/spm_disfluent', 
    vocab_size=1000, 
    model_type='unigram',
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=punc_list)

spm.SentencePieceTrainer.Train(
    input='original.txt', 
    model_prefix='../Datasets/Disfl-QA/spm_original', 
    vocab_size=1000, 
    model_type='unigram',
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=punc_list)

spm.SentencePieceTrainer.Train(
    input=glob.glob('*.txt'), 
    model_prefix='../Datasets/Disfl-QA/spm', 
    vocab_size=1000, 
    model_type='unigram',
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=punc_list)

# Test the model
sp_dis = spm.SentencePieceProcessor(model_file='../Datasets/DisFl-QA/spm_disfluent.model')
sp_ori = spm.SentencePieceProcessor(model_file='../Datasets/DisFl-QA/spm_original.model')
sp_all = spm.SentencePieceProcessor(model_file='../Datasets/DisFl-QA/spm.model')
enc = sp_all.Encode('what do unstable isotope studies indicate?')
sp_all.Decode(enc)