import openai
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

CATEGORY_DEFINITIONS = {
    'aspersion': 'Aspersion is the use of disrespectful attacks or damaging statements targeted towards ideas, plans, or policies.',
    'personal_attack': 'Personal attack is specific damaging or derogatory remarks towards participants in the conversation.',
    'third_party_attack': 'Third party attack is specific damaging or derogatory remarks towards others such as group of people, branch of the government, or political party.',
    'stereotype': 'Stereotype is the use of neutral or negative generalizations or labels, or impose discrimination upon certain groups.',
    'vulgarity': 'Vulgarity is the user of vulgar language or abbreviations of such with toxic intentions towards the discussion or fellow discussants.',
    'civility': 'Incivility are behaviors that disrespect towards individuals, groups, political communities, or topics of discussion.'
    }

def compute_metrics(y_pred, y_true):
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        roc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc = 0
    acc = accuracy_score(y_true, y_pred)
    return f1, roc, acc

def construct_category_header(category, k, examples_df):
    '''
    construct the header for a certain category to be used in prompts

    Parameters
    ----------
    category : str
    k : number of examples for each class
    examples_df : pandas dataframe (comment, 1/0 label)

    Returns
    -------
    header : str
    '''
    defn = CATEGORY_DEFINITIONS[category]
    header = f"The following is a list of comments and True/False labels of whether they contain {category}.\n\n"
    header = ' '.join((defn, header))
    # randomize example orders
    examples_df = examples_df.sample(frac=1)
    # avoid newlines inside comments
    comments = examples_df['comment'].apply(lambda x: x.replace('\n', '')).tolist()
    labels_str = examples_df[category].apply(lambda x: 'False' if x == 0 else 'True').tolist()
    examples_lst = list(zip(comments, labels_str))
    examples = [f"Comment: {comment}\nHas_{category}:{label}\n" for (comment, label) in examples_lst]
    examples_str = '###\n'.join(examples)
    header = ''.join((header, examples_str))
    return header
    
def construct_prompt(header, query):
    '''
    construct prompt for gpt-3

    Parameters
    ----------
    header : str, header for a fixed category
    query : str, test example

    Returns
    -------
    prompt : str

    '''
    query = query.replace('\n', '') # avoid newlines in comments
    query_str = f'Comment: {query}\nHas_{category}:'
    query_str = f'###\n{query_str}'
    prompt = ''.join((header, query_str))
    return prompt
    

openai.api_key = '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'
# Define the model to be used
COMPLETIONS_MODEL = "text-davinci-003"
# Define parameters
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 5,
    "model": COMPLETIONS_MODEL,
}



# read in data
file_path = './data/labeled/final_annotated_data_incivility_3030_processed.pickle'
annotated_df = pd.read_pickle(file_path)

# extract examples without any incivility, from which we'll sample neg examples
def no_incivility(row):
    return row['aspersion']==0 and row['namecalling']==0 \
            and row['stereotype']==0 and row['vulgarity']==0 \
            and row['other'] == 0 and row['human_incivility']==0
pure_negatives = annotated_df[annotated_df.apply(lambda r: no_incivility(r), axis=1)]

k = 5 # num of examples for each class (pos/neg)
seed=42
batchsize = 10
predictions = dict()
performances = dict()
for category in CATEGORY_DEFINITIONS.keys():
    # sample training examples
    neg_examples = pure_negatives.sample(n=k, random_state=seed)
    pos_df = annotated_df[annotated_df[category]!=0]
    pos_examples = pos_df.sample(n=k, random_state=seed)
    examples_df = pd.concat([pos_examples, neg_examples])
    training_ids = examples_df['id'].tolist()
    test_df = annotated_df[annotated_df['id'].apply(lambda x: x not in training_ids)]
    # generate header
    cat_header = construct_category_header(category, k, examples_df)
    test_comments = test_df['comment'].tolist()
    # make predictions with gpt3
    y_pred_cat_raw = []
    for i in range(0, len(test_comments), batchsize):
        query_lst = test_comments[i:i+batchsize]
        prompt_lst = [construct_prompt(cat_header, q) for q in query_lst]
        response = openai.Completion.create(
                        prompt=prompt_lst,
                        **COMPLETIONS_API_PARAMS
                    )
        response_lst = response['choices']
        y_pred = [obj['text'] for obj in response_lst]
        y_pred_cat_raw.extend(y_pred)
    # save predictions
    predictions[category] = y_pred_cat_raw
    # convert generated string to int
    y_pred_cat_int = [1 if x=='True' else 0 for x in y_pred_cat_raw]
    y_true_cat = test_df[category].tolist()
    # compute performance metrics
    f1, auc, acc = compute_metrics(y_pred_cat_int, y_true_cat)
    performances[category] = {'f1': f1, 'auc': auc, 'acc': acc}
    
