import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import datetime
import matplotlib.pyplot as plt
import torch
print(torch.__version__)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from torch.utils.data import TensorDataset,Subset, ConcatDataset
# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import seaborn as sns

import sys

def get_bert_model(device):  
    model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased", 
      num_labels = 2,           
      output_attentions = False, 
      output_hidden_states = True, 
    )
    # Tell pytorch to run this model on the GPU.
    model.to(device)
    return model

def get_model_auto(model_name, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 2,           
        output_attentions = False, 
        output_hidden_states = True, 
    )
    # Tell pytorch to run this model on the GPU.
    model.to(device)
    return model

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    try:
        roc = roc_auc_score(pred_flat, labels_flat)
    except ValueError:
        roc = 0
    return f1_score(pred_flat, labels_flat, average='macro'), precision_score(pred_flat, labels_flat, average='macro'), recall_score(pred_flat, labels_flat, average='macro'), roc, np.sum(pred_flat == labels_flat) / len(labels_flat)

def _flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def cross_validation(model, _X, _y, _cv=5):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": t['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }
    
def get_data_loaders(batch_size, dataset, train_indexes, val_indexes):
    return get_data_loader(batch_size, dataset, train_indexes),get_data_loader(batch_size, dataset, val_indexes)
  
def get_data_loader(batch_size, dataset, indexes='undefined'):
    if indexes != 'undefined':
        dataset = Subset(dataset, indexes)
    dataloader = DataLoader(
        dataset, 
        sampler = RandomSampler(dataset), 
        batch_size = batch_size
    )
    return dataloader

def get_augmented_training_set(dataset, aug_dataset, indexes):
    original = Subset(dataset, indexes)
    aug = Subset(aug_dataset, indexes)
    concat = torch.utils.data.ConcatDataset([original, aug])
    return concat

def draw_test_train_curve(test_losses, train_losses):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(test_losses, 'r-o', label='Test')

    # Label the plot.
    plt.title(f"Train/Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

def run_evaluation(model, test_x, test_labels, test_masks, test_type, batch_size, verbose=False):
    if verbose:
        print(f"{list(test_labels).count(1)} positive samples out of {len(test_labels)} total lines")
        print('Predicting labels for {:,} test sentences...'.format(len(test_x)))
    
    test_data = TensorDataset(test_x, test_masks, test_labels, test_type)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    # Put model in evaluation mode
    model.eval()
    
    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_type_ids = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=b_type_ids, 
                          attention_mask=b_input_mask)

        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        
    # Create results
    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    if verbose:
        print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        # The predictions for this batch are a 2-column ndarray (one column for "0" 
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        # Calculate and store the coef for this batch.  
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        
        if verbose:
            print("Predicted Label for Batch " + str(i) + " is " + str(pred_labels_i))
            print("True Label for Batch " + str(i) + " is " + str(true_labels[i])) 
            print("Matthew's correlation coefficient for Batch " + str(i) + " is " + str(matthews))
        matthews_set.append(matthews)
    
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    diff = []
    for i in range(len(flat_true_labels)):
      if flat_true_labels[i] != flat_predictions[i]:
        diff.append(i)

    # Calculate the MCC
    acc = accuracy_score(flat_predictions, flat_true_labels)
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
    ra = roc_auc_score(flat_true_labels, flat_predictions)

    cm = confusion_matrix(flat_true_labels, flat_predictions)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

    print('MCC: %.3f' % mcc)
    print('ROC_AUC: %.3f' % ra)
    print('F1: %.3f' % f1)
    print('Accuracy: %.3f' % acc)
    print(classification_report(flat_true_labels, flat_predictions))

    return diff

def column_to_tensor(df, column_name):
    return torch.tensor(df[column_name].values.tolist())