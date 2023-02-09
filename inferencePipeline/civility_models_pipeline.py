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
from transformers import BertForSequenceClassification


def get_discourse_acts(loader, path, batch_size, label):
    with torch.no_grad():
        fname = path + "/new_labels_{}.npy".format(label)
        if os.path.isfile(fname):
          cs = list(np.load(fname))
        else:
          cs = []
        start = len(cs) // batch_size
        for i, batch in enumerate(loader):
            if i >= start:
              if i % 2000 == 0:
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

categories = ['stereotype']
batch_size = 128
path = sys.argv[1]

# use GPU or CPU as device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

dataset = Dataset.load_from_disk(path)
dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

output_dir = './models/stereotype_model_downsampling'
model = BertForSequenceClassification.from_pretrained(output_dir)
model = model.to(device)