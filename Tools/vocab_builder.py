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
spm.SentencePieceTrainer.Train(input=glob.glob('*.txt'), model_prefix='../Datasets/Disfl-QA/spm', vocab_size=5000, model_type='unigram')

# Test the model
sp = spm.SentencePieceProcessor(model_file='../Datasets/DisFl-QA/spm.model')
enc = sp.Encode('what do unstable isotope studies indicate?')
sp.Decode(enc)