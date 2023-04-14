import torch
print(torch.__version__)
import pandas as pd
import time
from torch.utils.data import TensorDataset
from transformers import AdamW
from sklearn.model_selection import StratifiedKFold
import gc
import os
import json
import argparse
import sys

# nohup python -u model_trainings_all_data.py --model roberta --gpu 0 --category civility --lr 8e-6   > output_all_training.log &
# nohup python -u model_trainings_roberta.py --model bert > output_bert_aug.log &

from model_utils import get_model_auto, get_data_loaders, format_time, flat_accuracy, column_to_tensor, get_data_loader, get_augmented_training_set

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Please enter model, now supported are bert and roberta', default='bert')
parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--lr', type=float, default=1e-5) # lrs = [8e-6, 1e-5] 
args = parser.parse_args()

model_type = args.model
# categories=['aspersion', 'civility', 'vulgarity', 'personal_attack', 'third_party_attack', 'stereotype']
categories = [args.category]
lr = args.lr

if model_type == 'bert':
    model_name = 'bert-base-uncased'
    annotated_data_dir = './data/labeled/final_annotated_data_incivility_3030_processed.pickle'
elif model_type == 'roberta':    
    model_name = 'roberta-base'
    annotated_data_dir = './data/labeled/final_annotated_data_roberta.pickle'
else:
    print("Your selected model is not supported yet")
    sys.exit()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(1))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
annotated_df = pd.read_pickle(annotated_data_dir)
print(annotated_df.head())
batch_size=8
use_augmentation = False

for category in categories:
    print("start training for category {} at learning rate {}".format(category, lr))
    gc.collect()
    torch.cuda.empty_cache()

    epochs = 5

    labels = annotated_df[category].values.astype(int).astype(bool).astype(int).tolist()
    if model_name.startswith("bert"):
        dataset = TensorDataset(column_to_tensor(annotated_df, 'bert'), column_to_tensor(annotated_df, 'attention'), torch.tensor(labels))
        aug_dataset = TensorDataset(column_to_tensor(annotated_df, 'bert_aug'), column_to_tensor(annotated_df, 'attention_aug'), torch.tensor(labels))
    elif model_name.startswith("roberta"):
        dataset = TensorDataset(column_to_tensor(annotated_df, 'roberta'), column_to_tensor(annotated_df, 'attention_mask'), torch.tensor(labels))
        aug_dataset = TensorDataset(column_to_tensor(annotated_df, 'roberta_aug'), column_to_tensor(annotated_df, 'attention_mask_aug'), torch.tensor(labels))

    model = get_model_auto(model_name, device)
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    if use_augmentation:
        train_aug_sets = get_augmented_training_set(dataset, aug_dataset, list(range(len(dataset))))
        train_dataloader = get_data_loader(batch_size, train_aug_sets)
    else:
        train_dataloader = get_data_loader(batch_size, dataset)
    for epoch_i in range(0, epochs):
        print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):

            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed:}.')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        

            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)


            total_train_loss += outputs.loss.item()

            # Perform a backward pass to calculate the gradients.
            outputs.loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #update weights
            optimizer.step()


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        save_dir = f'checkpoints/{model_type}_ckpts_final/{category}'
        subdir = f"lr{lr}"
        save_path = os.path.join(save_dir, subdir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(f"{save_path}/epoch_{epoch_i}.pt")

    print("------finished training for category {} at learning rate {}------".format(category, lr))
