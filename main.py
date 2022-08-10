import prefect
from prefect import task, Flow
from models import search_engine, topic_modelling
import os

PATH_REPOS_CLEAN = "http://openpharma.s3-website.us-east-2.amazonaws.com/repos_clean.csv"
PATH_SEARCH_TEST = "http://openpharma.s3-website.us-east-2.amazonaws.com/ml/query_test_set.csv"
PATH_OPEN_ISSUES = "https://openpharma.s3.us-east-2.amazonaws.com/help_clean.csv"
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('OPENPHARMA_AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('OPENPHARMA_AWS_SECRET_ACCESS_KEY')


with Flow("Search-engine Packages") as flow1:
    X, Y, X_test, Y_test = search_engine.packages_read_csv(path_repos=PATH_REPOS_CLEAN, path_test=PATH_SEARCH_TEST)
    X_clean = search_engine.clean_data(X, is_lemma=False, remove_stop=False, is_alphabetic=True)
    X_test_clean = search_engine.clean_data(X_test, is_lemma=False, remove_stop=False, is_alphabetic=True)
    X_embed = search_engine.inference_pretrained(X_clean)
    X_test_embed = search_engine.inference_pretrained(X_test_clean)

    search_engine.save_model(X_embed, key_id=os.getenv('AWS_ACCESS_KEY_ID'), access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), file_name="inference_packages.pt")
flow1.run()

with Flow("Search-engine Open issues") as flow2:
    X = search_engine.openissues_read_csv(path_openissues=PATH_OPEN_ISSUES)
    X_clean = search_engine.clean_data(X, is_lemma=False, remove_stop=False, is_alphabetic=True)
    X_embed = search_engine.inference_pretrained(X_clean)
    search_engine.save_model(X_embed, key_id=os.getenv('AWS_ACCESS_KEY_ID'), access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), file_name="inference_openissues.pt")
flow2.run()