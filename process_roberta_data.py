from transformers import AutoTokenizer
import pandas as pd
import argparse
import sys

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

tokenizer = AutoTokenizer.from_pretrained(model_name)
annotated_df = pd.read_pickle(annotated_data_dir)

if model_type == 'roberta':
    if 'roberta' not in annotated_df.columns:
        encodings = tokenizer.batch_encode_plus(annotated_df.comment.tolist(), max_length=512, truncation=True, padding='max_length')
        annotated_df['roberta'] = encodings['input_ids']
        annotated_df['attention_mask'] = encodings['attention_mask']
    
    #augmentation
    print('augmentation')
    encodings = tokenizer.batch_encode_plus(annotated_df.comment_en.tolist(), max_length=512, truncation=True, padding='max_length')
    annotated_df['roberta_aug'] = encodings['input_ids']
    annotated_df['attention_mask_aug'] = encodings['attention_mask']
elif model_type == 'bert':
    if 'bert' not in annotated_df.columns:
        encodings = tokenizer.batch_encode_plus(annotated_df.comment.tolist(), max_length=512, truncation=True, padding='max_length')
        annotated_df['roberta'] = encodings['input_ids']
        annotated_df['attention_mask'] = encodings['attention_mask']
    
    #augmentation
    print('augmentation')
    encodings = tokenizer.batch_encode_plus(annotated_df.comment_en.tolist(), max_length=512, truncation=True, padding='max_length')
    annotated_df['bert_aug'] = encodings['input_ids']
    annotated_df['attention_aug'] = encodings['attention_mask']
annotated_df.to_pickle(annotated_data_dir)
