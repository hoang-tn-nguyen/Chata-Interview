import json
import spacy
import sentencepiece as spm
json_file = open('../Datasets/Disfl-QA/train.json')
data = json.load(json_file)

original_file = open('original.txt','w',encoding='utf-8')
disfluent_file = open('disfluent.txt','w',encoding='utf-8')

# Extract tokens & frequency from sentences
my_dict = {}
for k,v in data.items():
    original = v['original'].lower().split()
    disfluent = v['disfluent'].lower().split()

    original_file.write(v['original'].lower() + '\n')
    disfluent_file.write(v['disfluent'].lower() + '\n')

    for word in original:
        if word in my_dict:
            my_dict[word] += 1
        else:
            my_dict[word] = 1

    for word in disfluent:
        if word in my_dict:
            my_dict[word] += 1
        else:
            my_dict[word] = 1

original_file.close()
disfluent_file.close()

# Use spacy to post process the tokens e.g., "warsaw?" --> "warsaw" and "?".
nlp = spacy.load("en_core_web_sm")

vocab = {}
for k,v in my_dict.items():
    out = nlp(k)
    for tok in out:
        if tok not in vocab:
            vocab[tok] = v
        else:
            vocab[tok] += v

vocab_list = sorted([(k,v) for k,v in vocab.items()], key=lambda x: x[1],reverse=True)
vocab_list = [str(item[0]) for item in vocab_list][:1000]

# Build vocabulary with sentencepiece
import glob
spm.SentencePieceTrainer.Train(input=glob.glob('*.txt'), model_prefix='../Datasets/Disfl-QA/spm', vocab_size=2000, model_type='unigram')
sp = spm.SentencePieceProcessor(model_file='../Datasets/DisFl-QA/spm.model')