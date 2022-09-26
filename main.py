import prefect
from prefect import task, Flow
from models import package_categorization, search_engine
import os

KEYWORDS = "category_keywords.json"
PATH_REPOS_CLEAN = "http://openpharma.s3-website.us-east-2.amazonaws.com/repos_clean.csv"
PATH_ANNOTATED = "https://openpharma.s3.us-east-2.amazonaws.com/ml/repos_annotated.csv"
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


with Flow("Search-engine Open issues") as flow2:
    X = search_engine.openissues_read_csv(path_openissues=PATH_OPEN_ISSUES)
    X_clean = search_engine.clean_data(X, is_lemma=False, remove_stop=False, is_alphabetic=True)
    X_embed = search_engine.inference_pretrained(X_clean)
    search_engine.save_model(X_embed, key_id=os.getenv('AWS_ACCESS_KEY_ID'), access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), file_name="inference_openissues.pt")


with Flow("Package Categorization") as flow3:
    df_anno = package_categorization.read_csv(path_csv=PATH_ANNOTATED)
    df_repos = package_categorization.read_csv(path_csv=PATH_REPOS_CLEAN)
    d = package_categorization.read_json(file_path=KEYWORDS)
    # Data cleaning and prediction
    df_repos = package_categorization.clean_data(df=df_repos)
    df_repos = package_categorization.predict_category(df=df_repos, d=d)
    # Merge inference and already annotated
    df_repos = package_categorization.merge_repos_annotated(df_repos=df_repos, df_anno=df_anno)
    package_categorization.save_dataframe(df=df_repos, key_id=os.getenv('AWS_ACCESS_KEY_ID'), access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), file_name="repos_categorization.csv")


flow1.run()
flow2.run()
flow3.run()