import pandas as pd
from typing import Dict, List
import json
import spacy
import ast
from prefect import task
import boto3

@task
def read_csv(path_csv: str):
    df = pd.read_csv(path_csv)
    return df

@task
def read_json(file_path: str)-> Dict:
    with open(file_path, "r") as fi:
        d = json.loads(fi.read())
    return d


def clean_text_nlp(X, is_lemma=True, remove_stop=True, is_alphabetic=True):
    """
    X : list of string such as ["sentence_1", "sentences_2", ... , "sentence_n"]
    Return : list of list of words 
    [
        ["word_1", "word_2", ... , "word_n"], (sentence 1 cleaned)
        ["word_1", "word_2", ... , "word_n"],
        .
        .
        .
        ["word_1", "word_2", ... , "word_n"] (sentence n cleaned)
    ]
    """
    nlp = spacy.load("en_core_web_sm")
    # (is_lemma = True, remove_stop = True, is_alpha = False) = (1,1,0)
    hyperparam_tuple = (is_lemma, remove_stop, is_alphabetic)
    new_X = []
    while(True):
        if((0,0,0)==hyperparam_tuple): # everything initialize with False
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.text.lower() for token in doc if(not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(doc)
            break
        elif((1,0,0)==hyperparam_tuple): # lemma_
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.lemma_.lower() for token in doc if(not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(doc)
            break
        elif((0,1,0)==hyperparam_tuple): # remove stop_words
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.text.lower() for token in doc if (not(token.is_stop) and not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(doc)
            break
        elif((0,0,1)==hyperparam_tuple): # only alphabetic
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.text.lower() for token in doc if (token.is_alpha and not(token.is_left_punct))]
                new_X.append(doc)
            break
        elif((1,1,0)==hyperparam_tuple): #lemma and remove stop_words
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.lemma_.lower() for token in doc if (not(token.is_stop) and not(token.is_left_punct) and not(token.is_right_punct) and not(token.is_punct) and not(token.is_bracket))]
                new_X.append(doc)
            break
        elif((0,1,1)==hyperparam_tuple): #remove stop_words and only alphabetic
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.text.lower() for token in doc if (token.is_alpha and not(token.is_stop))]
                new_X.append(doc)
            break
        elif((1,0,1)==hyperparam_tuple): #lemma and only alphabetic
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.lemma_.lower() for token in doc if(token.is_alpha and not(token.is_left_punct))]
                new_X.append(doc)
            break
        elif((1,1,1)==hyperparam_tuple): #lemma and only alphabetic and remove stop_words
            for doc in nlp.pipe(X, disable=['ner', 'parser', 'textcat']):
                doc = [token.lemma_.lower() for token in doc if(token.is_alpha and not(token.is_stop))]
                new_X.append(doc)
            break
        break

    return new_X

@task
def clean_data(df: pd.DataFrame)-> pd.DataFrame:
    df["description"].fillna("", inplace=True)
    df["title"].fillna("", inplace=True)
    df["description"] = df["title"] + " " + df["description"]
    X = df["description"].to_list()
    X_clean = clean_text_nlp(X, is_lemma=True, remove_stop=True, is_alphabetic=True)
    df["description"] = X_clean
    return df


def binary_assignement(x: str, l: List)-> int:
    return True if 1 in [1 if w in l else 0 for w in x] else False

@task
def predict_category(df: pd.DataFrame, d: Dict)-> pd.DataFrame:
    df["plots_inf"] = df["description"].apply(lambda x: binary_assignement(x=x, l=ast.literal_eval(d["plots"])))
    df["tables_inf"] = df["description"].apply(lambda x: binary_assignement(x=x, l=ast.literal_eval(d["tables"])))
    df["stats_inf"] = df["description"].apply(lambda x: binary_assignement(x=x, l=ast.literal_eval(d["stats"])))
    df["cdisc_inf"] = df["description"].apply(lambda x: binary_assignement(x=x, l=ast.literal_eval(d["cdisc"])))
    df["utilities_inf"] = df["description"].apply(lambda x: binary_assignement(x=x, l=ast.literal_eval(d["utilities"])))
    return df

@task
def merge_repos_annotated(df_repos: pd.DataFrame, df_anno: pd.DataFrame)-> pd.DataFrame:
    df_repos = df_repos.merge(df_anno, how="left", on="full_name")
    df_repos["plots"].fillna(df_repos["plots_inf"], inplace=True)
    df_repos["tables"].fillna(df_repos["tables_inf"], inplace=True)
    df_repos["stats"].fillna(df_repos["stats_inf"], inplace=True)
    df_repos["cdisc"].fillna(df_repos["cdisc_inf"], inplace=True)
    df_repos["utilities"].fillna(df_repos["utilities_inf"], inplace=True)
    df_repos = df_repos[["full_name", "plots", "tables", "stats", "cdisc", "utilities"]]
    return df_repos

@task
def save_dataframe(df: pd.DataFrame, key_id: str, access_key: str, file_name: str):
    df.to_csv(file_name, index=False)
    client = boto3.client('s3',
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key
    )
    client.upload_file(Filename=file_name,
        Bucket='openpharma',
        Key='ml/{}'.format(file_name)
    )
    return 0