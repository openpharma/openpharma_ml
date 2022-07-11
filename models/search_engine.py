import prefect
from prefect import task, Flow
import pandas as pd
import spacy

PATH_REPOS_CLEAN = "http://openpharma.s3-website.us-east-2.amazonaws.com/repos_clean.csv"
PATH_TEST_BASE = "http://openpharma.s3-website.us-east-2.amazonaws.com/search_bar_test_base.csv"

@task
def download_data(path: str):
    df = pd.read_csv(PATH_REPOS_CLEAN)
    df_test  = pd.read_csv(PATH_TEST_BASE)
    X = df['description'].to_list()
    X_test = df_test
    return X, X_test
    
@task
def clean_data(X, is_lemma=True, remove_stop=True, is_alphabetic=True):
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
def inference_pretrained():

    #embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #embed_corpus = embedder.encode(new_X, convert_to_tensor=True)
    return 0

@task
def scoring(X_vector, X_test_vector):

    #embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #embed_corpus = embedder.encode(new_X, convert_to_tensor=True)
    return 0

@task
def save_model(X_vector):
    #from X_vector tensor to S3 bucket /ML directory
    return 0