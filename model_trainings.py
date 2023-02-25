import torch
print(torch.__version__)
import pandas as pd
import time
from torch.utils.data import TensorDataset
from transformers import AdamW
from sklearn.model_selection import StratifiedKFold
import gc

# nohup python -u  model_trainings.py > output.log &

from model_utils import get_bert_model, get_data_loaders, format_time, flat_accuracy, column_to_tensor

if torch.cuda.is_available():
    device = torch.device("cuda:0")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(1))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

categories = ['aspersion', 'vulgarity', 'personal_attack', 'third_party_attack', 'civility', 'stereotype']
lrs = [5e-6, 8e-6, 1e-5, 2e-5, 5e-05]
annotated_data_dir = './data/labeled/final_annotated_data_incivility_3030_processed.pickle'
annotated_df = pd.read_pickle(annotated_data_dir)
print(annotated_df.head())
batch_size=8


for category in categories:
  for lr in lrs:
    print("start training for category {} at learning rate {}".format(category, lr))
    gc.collect()
    torch.cuda.empty_cache()
    
    total_folds = 2
    current_fold = 0
    all_folds_preds = []
    epochs = 5
    
    labels = annotated_df[category].values.astype(int).astype(bool).astype(int).tolist()
    dataset = TensorDataset(column_to_tensor(annotated_df, 'bert'), column_to_tensor(annotated_df, 'attention'), torch.tensor(labels), column_to_tensor(annotated_df, 'type_id'))

    fold=StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=1000)
    training_stats = []

    for train_index, test_index in fold.split(annotated_df, annotated_df[category]):
        model = get_bert_model()
        optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
        current_fold = current_fold+1
        print('\n================= Fold {:} / {:} ================='.format(current_fold, total_folds))
        train_dataloader,validation_dataloader = get_data_loaders(batch_size, dataset, train_index, test_index)
        
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
                tmp_eval_f1, tmp_eval_auc, tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                total_eval_accuracy += tmp_eval_accuracy
                total_f1_score += tmp_eval_f1
                total_auc += tmp_eval_auc

            # Report the final accuracy and f1_score for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            
            avg_f1_score = total_f1_score / len(validation_dataloader)
            print("  F1_score: {0:.2f}".format(avg_f1_score))

            avg_auc_score = total_auc / len(validation_dataloader)
            print("  AUC_score: {0:.2f}".format(avg_auc_score))
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
            {
                'category': category,
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'f1_score' : avg_f1_score,
                'AUC_score': avg_auc_score,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'fold' : current_fold,
                'learning_rate': lr
            }
            )

    print(training_stats)


