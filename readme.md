# A Crisis of Civility? Modeling Incivility and Its Effects in Political Discourse Online

*Yujia Gao, Wenna Qin, Aniruddha Murali, Christopher Eckart, Xuhui Zhou, Jacob Daniel Beel, Yi-Chia Wang, Diyi Yang*


## Data

As this work studies the sensitive domain of political discussions, we only release the comment ID and annotations while leaving out the comment texts themselves to at least ensure that users who might want to edit or delete their statements about personal preferences after the time of our scrape.

## Instructions

To set up the environment:
```
conda create --name incivility python=3.9
conda activate incivility
pip install -r requirements.txt
```


To train BERT/RoBERTa models:
```
python -u model_trainings_roberta.py --model roberta
```