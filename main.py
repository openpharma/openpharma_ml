import prefect
from prefect import task, Flow
from models import search_engine, topic_modelling


PATH_REPOS_CLEAN = "http://openpharma.s3-website.us-east-2.amazonaws.com/repos_clean.csv"
PATH_SEARCH_TEST = "http://openpharma.s3-website.us-east-2.amazonaws.com/ml/query_test_set.csv"

with Flow("Search-engine") as flow1:
    X, Y, X_test, Y_test = search_engine.download_data(path_repos=PATH_REPOS_CLEAN, path_test=PATH_SEARCH_TEST)
    X_clean = search_engine.clean_data(X, is_lemma=False, remove_stop=True, is_alphabetic=True)
    X_test_clean = search_engine.clean_data(X_test, is_lemma=False, remove_stop=True, is_alphabetic=True)
    X_embed = search_engine.inference_pretrained(X_clean)
    X_test_embed = search_engine.inference_pretrained(X_test_clean)

    search_engine.save_model(X_embed, user='', password='')
flow1.run()