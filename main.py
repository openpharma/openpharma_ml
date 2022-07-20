import prefect
from prefect import task, Flow
from models import search_engine, topic_modelling
import os

PATH_REPOS_CLEAN = "http://openpharma.s3-website.us-east-2.amazonaws.com/repos_clean.csv"
PATH_SEARCH_TEST = "http://openpharma.s3-website.us-east-2.amazonaws.com/ml/query_test_set.csv"
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('OPENPHARMA_AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('OPENPHARMA_AWS_SECRET_ACCESS_KEY')


with Flow("Search-engine") as flow1:
    X, Y, X_test, Y_test = search_engine.download_data(path_repos=PATH_REPOS_CLEAN, path_test=PATH_SEARCH_TEST)
    X_clean = search_engine.clean_data(X, is_lemma=False, remove_stop=False, is_alphabetic=True)
    X_test_clean = search_engine.clean_data(X_test, is_lemma=False, remove_stop=False, is_alphabetic=True)
    X_embed = search_engine.inference_pretrained(X_clean)
    X_test_embed = search_engine.inference_pretrained(X_test_clean)

    search_engine.save_model(X_embed, key_id=os.getenv('AWS_ACCESS_KEY_ID'), access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
flow1.run()