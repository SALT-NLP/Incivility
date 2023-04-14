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

# nohup python -u model_trainings_roberta.py --model roberta > output_roberta_aug.log &
# nohup python -u model_trainings_roberta.py --model bert > output_bert_aug.log &

from model_utils import get_model_auto, get_data_loaders, format_time, flat_accuracy, column_to_tensor, get_data_loader, get_augmented_training_set

if torch.cuda.is_available():
    device = torch.device("cuda:1")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(1))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Please enter model, now supported are bert and roberta', default='bert')
args = parser.parse_args()
model_type = args.model
if model_type == 'bert':
    model_name = 'bert-base-uncased'
    annotated_data_dir = './data/labeled/final_annotated_data_incivility_3030_processed.pickle'
elif model_type == 'roberta':    
    model_name = 'roberta-base'
    annotated_data_dir = './data/labeled/final_annotated_data_roberta.pickle'
else:
    print("Your selected model is not supported yet")
    sys.exit()
    
# categories = ['aspersion', 'civility', 'vulgarity', 'personal_attack', 'third_party_attack', 'stereotype']
# lrs = [5e-6, 8e-6, 1e-5, 2e-5, 5e-05] 
categories = ['aspersion', 'civility', 'vulgarity', 'personal_attack', 'third_party_attack', 'stereotype']
lrs = [8e-6, 1e-5] 
annotated_df = pd.read_pickle(annotated_data_dir)
print(annotated_df.head())
batch_size=8
use_augmentation = True

all_training_stats = []

for category in categories:
  for lr in lrs:
    print("start training for category {} at learning rate {}".format(category, lr))
    gc.collect()
    torch.cuda.empty_cache()
    
    total_folds = 5
    current_fold = 0
    epochs = 5
    
    labels = annotated_df[category].values.astype(int).astype(bool).astype(int).tolist()
    if model_name.startswith("bert"):
        dataset = TensorDataset(column_to_tensor(annotated_df, 'bert'), column_to_tensor(annotated_df, 'attention'), torch.tensor(labels))
        aug_dataset = TensorDataset(column_to_tensor(annotated_df, 'bert_aug'), column_to_tensor(annotated_df, 'attention_aug'), torch.tensor(labels))
    elif model_name.startswith("roberta"):
        dataset = TensorDataset(column_to_tensor(annotated_df, 'roberta'), column_to_tensor(annotated_df, 'attention_mask'), torch.tensor(labels))
        aug_dataset = TensorDataset(column_to_tensor(annotated_df, 'roberta_aug'), column_to_tensor(annotated_df, 'attention_mask_aug'), torch.tensor(labels))
    fold=StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=1000)
    training_stats = []

    for train_index, test_index in fold.split(annotated_df, annotated_df[category]):
        # keep track of the best model so far
        best_val_acc = 0
        # model = get_bert_model()
        model = get_model_auto(model_name, device)
        optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
        current_fold = current_fold+1
        print('\n================= Fold {:} / {:} ================='.format(current_fold, total_folds))
        if use_augmentation:
            train_aug_sets = get_augmented_training_set(dataset, aug_dataset, train_index)
            validation_dataloader = get_data_loader(batch_size, dataset, test_index)
            train_dataloader = get_data_loader(batch_size, train_aug_sets)
        else:
            train_dataloader,validation_dataloader = get_data_loaders(batch_size, dataset, train_index, test_index)
        print(len(train_dataloader), len(validation_dataloader))
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

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            model.eval()

            # Tracking variables 
            total_f1_score = 0
            total_auc = 0
            total_precision = 0
            total_recall = 0
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():        
                    outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                # Accumulate the validation loss.
                total_eval_loss += outputs.loss.item()

                # Move logits and labels to CPU
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                tmp_eval_f1, tmp_eval_precision, tmp_eval_recall, tmp_eval_auc, tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                total_eval_accuracy += tmp_eval_accuracy
                total_precision += tmp_eval_precision
                total_recall += tmp_eval_recall
                total_f1_score += tmp_eval_f1
                total_auc += tmp_eval_auc

            # Report the final accuracy and f1_score for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            
            avg_f1_score = total_f1_score / len(validation_dataloader)
            print("  F1_score: {0:.2f}".format(avg_f1_score))

            avg_precision_score = total_precision / len(validation_dataloader)
            print("  Precision: {0:.2f}".format(avg_precision_score))
            
            avg_recall_score = total_recall / len(validation_dataloader)
            print("  Recall: {0:.2f}".format(avg_recall_score))
            
            avg_auc_score = total_auc / len(validation_dataloader)
            print("  AUC_score: {0:.2f}".format(avg_auc_score))
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            if avg_val_accuracy > best_val_acc:
                best_val_acc = avg_val_accuracy
                save_dir = f'checkpoints/{model_type}_ckpts/{category}_aug'
                subdir = f"fold{current_fold}_lr{lr}"
                save_path = os.path.join(save_dir, subdir)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model.save_pretrained(f"{save_path}/best_model.pt")
                print(f"saved best model so far to {save_path}.")

            # Record all statistics from this epoch.
            training_stats.append(
            {
                'category': category,
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'f1_score' : avg_f1_score,
                'recall_score': avg_recall_score,
                'precision_score': avg_precision_score,
                'AUC_score': avg_auc_score,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'fold' : current_fold,
                'learning_rate': lr,
                'model': model_type
            }
            )
    print("------finished training for category {} at learning rate {}------".format(category, lr))
    print(training_stats)
    
        
    s = f'checkpoints/{model_type}_ckpts/{category}_aug'
    if not os.path.exists(s):
        os.makedirs(s)
    with open(os.path.join(s, f'training_stats_{lr}_with_precision_aug.json'), "w") as file:
        json.dump(training_stats, file, indent=0)

    all_training_stats.extend(training_stats)
    
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=all_training_stats)
    df_stats.to_pickle(f'checkpoints/{model_type}_ckpts/all_training_stats_aug.pickle')
