import prefect
from prefect import task
import boto3
import pandas as pd

"""
Input : X -> vector of decription text for each package
Output : Y = [0 0 1 ... 1 0] -> multilabel binarizer : 1 if in the category 0 otherwise

Metric : Accuracy, Precision, recall, F1 Score, confusion matrix, ROC Curve


Notebook step by step

#1. Read 2 dataset -> annotated data and df_repos_clean
#2. Clean annotated data and merge annotated with df_repos_clean on full_name
#3. Data cleaning :
    #3.1 Spacy pipeline to clean description
    #3.2 Multilabel Binarizer for Y label
#4. Fit Gensim model : Doc2Vec
#5. Create 2 columns : X and Y
#6. Use stratified splits because of class imbalance
#7. Fit Logistic regression : multilabel classification problem.
#8. Assess performance

"""