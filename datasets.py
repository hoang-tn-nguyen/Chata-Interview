import os
import json
import numpy as np
import pandas as pd
import sentencepiece as spm

import torch
import torch.utils.data as data

class DisflQA(data.Dataset):
    def __init__(self, file_name='Datasets/Disfl-QA/train.json'):
        self.__input_data(file_name)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['disfluent'], self.data[index]['original']

    def __input_data(self, file_name):
        try:
            json_file = open(file_name, 'r')
            data = json.load(json_file)
            self.data = [v for k,v in data.items()]
            json_file.close()

        except Exception as e:
            print('Error occurred:', e)




