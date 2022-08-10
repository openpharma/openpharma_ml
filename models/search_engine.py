import prefect
from prefect import task
import boto3
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import spacy
from typing import List


@task(nout=4)
def packages_read_csv(path_repos: str, path_test: str):
    df = pd.read_csv(path_repos)
    df_test  = pd.read_csv(path_test)
    X = df["description"].to_list()
    Y = df["full_name"].to_list()
    X_test = df_test["search query"].to_list()
    Y_test = df_test["packages"].to_list()
    return X, Y, X_test, Y_test

@task
def openissues_read_csv(path_openissues: str):
    df = pd.read_csv(path_openissues)
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text_to_vec"] = df["body"]+df["title"]
    X = df["text_to_vec"].to_list()
    return X
    
@task
def clean_data(X, is_lemma: bool=True, remove_stop: bool=True, is_alphabetic: bool=True):
    """
    X : list of string such as ["sentence_1", "sentences_2", ... , "sentence_n"]
    Return : list of list of words 
    [
        ["word_1 word_2 ... word_n"], (sentence 1 cleaned)
        ["word_1 ... word_n"],
        .
        .
        .
        ["word_1 word_2 ... word_n"] (sentence n cleaned)
    ]
    """
    nlp = spacy.load("en_core_web_sm")
    # (is_lemma = True, remove_stop = True, is_alpha = False) = (1,1,0)
    hyperparam_tuple = (is_lemma, remove_stop, is_alphabetic)
    new_X = []
    while(True):
        if((0,0,0)==hyperparam_tuple): # everything initialize with False
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.text.lower() for token in doc if(not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(" ".join(sentence))
            break
        elif((1,0,0)==hyperparam_tuple): # lemma_
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.lemma_.lower() for token in doc if(not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(" ".join(sentence))
            break
        elif((0,1,0)==hyperparam_tuple): # remove stop_words
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.text.lower() for token in doc if (not(token.is_stop) and not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(" ".join(sentence))
            break
        elif((0,0,1)==hyperparam_tuple): # only alphabetic
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.text.lower() for token in doc if (token.is_alpha and not(token.is_left_punct))]
                new_X.append(" ".join(sentence))
            break
        elif((1,1,0)==hyperparam_tuple): #lemma and remove stop_words
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.lemma_.lower() for token in doc if (not(token.is_stop) and not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(" ".join(sentence))
            break
        elif((0,1,1)==hyperparam_tuple): #remove stop_words and only alphabetic
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.text.lower() for token in doc if (token.is_alpha and not(token.is_stop))]
                new_X.append(" ".join(sentence))
            break
        elif((1,0,1)==hyperparam_tuple): #lemma and only alphabetic
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.lemma_.lower() for token in doc if(token.is_alpha and not(token.is_left_punct))]
                new_X.append(" ".join(sentence))
            break
        elif((1,1,1)==hyperparam_tuple): #lemma and only alphabetic and remove stop_words
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                sentence = [token.lemma_.lower() for token in doc if(token.is_alpha and not(token.is_stop))]
                new_X.append(" ".join(sentence))
            break
        break
    
    return new_X

@task
def inference_pretrained(X: List[str], model_name: str='BERT'):
    if(model_name=='BERT'):
        embedder = SentenceTransformer('LM-L6-BERT')
        embed_corpus = embedder.encode(X, convert_to_tensor=True)
    else:
        embed_corpus = 0
    return embed_corpus

@task
def scoring(X_vector, X_test_vector):
    # Scoring logic
    return 0

@task
def save_model(X_vector, key_id: str, access_key: str, file_name: str):
    torch.save(X_vector, file_name)
    client = boto3.client(
        's3',
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key
    )

    client.upload_file(Filename=file_name,
        Bucket='openpharma',
        Key='ml/{}'.format(file_name)
    )
    return 0