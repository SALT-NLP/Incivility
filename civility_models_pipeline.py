import os
import numpy as np
from datetime import date
import datetime
import torch
import time
import sys
import tqdm
from scipy.special import softmax
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification
from time import gmtime, strftime

# nohup python -u civility_models_pipeline.py data/datasets/20200315_20200528_all_comments_with_labels_first_roberta 0 > 20200315_20200528_first.log &
# nohup python -u civility_models_pipeline.py data/datasets/20200315_20200528_all_comments_with_labels_second_roberta 1 > 20200315_20200528_seonc.log &

def get_discourse_acts(loader, path, batch_size, label):
    with torch.no_grad():
        fname = path + "/{}.npy".format(label)
        if os.path.isfile(fname):
          cs = list(np.load(fname))
        else:
          cs = []
        start = len(cs) // batch_size
        print("total ", len(loader), "starting from ", start)
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        for i, batch in enumerate(loader):
            if i >= start:
              if i != 0 and i % 10 == 0:
                print(i)
                np.save(fname, cs)
              ct = batch['input_ids']
              cm = batch['attention_mask']
              ct = ct.to(device)
              cm = cm.to(device)
              c_act = model(ct, cm)[0].cpu().numpy()
              for c in c_act:
                  cs.append(c)
        np.save(fname, cs)              
    return cs

def text_to_discourse_acts(loader, dataset, label, path, batch_size):
    print("Vectorizing threads...")
    print("Running BERT")
    c_acts = get_discourse_acts(loader, path, batch_size, label)
    print("Assigning acts...")
    return c_acts


categories = ['civility', 'third_party_attack', 'aspersion', 'stereotype', 'vulgarity', 'personal_attack']
# categories = ['third_party_attack']
batch_size = 256
path = sys.argv[1]
gpu_id = sys.argv[2]

dataset = Dataset.load_from_disk(path)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# use GPU or CPU as device
if torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_id}')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

for category in categories:
    print(category)
    output_dir = './final_models/{}'.format(category)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model = model.to(device)
    text_to_discourse_acts(dataloader, dataset, category, path, batch_size)