# Openpharma : ML for search bar and data categorization <img src="logo.png" align="right" width="200" style="margin-left:50px;"/>
The objective of openpharma is to provide a neutral home for open source software related to pharmaceutical industry that is not tied to one company or institution. http://openpharma.pharmaverse.org/

📨 For any questions, feel free to reach me out at the email adress : mathieu.cayssol@gmail.com

# 0. General overview

## Global pipeline

You are in the front-end repository of openpharma. The global project include 3 repositories :
 - ⚙️ Data crawler : https://github.com/openpharma/openpharma.github.io
 - 🤖 ML for search bar and data categorization : https://github.com/openpharma/openpharma_ml
 - 📊 Front-end : https://github.com/openpharma/opensource_dashboard


# 1. Search bar Pipeline

<img width="1550" alt="Pipeline_search_bar" src="https://user-images.githubusercontent.com/49449000/191011621-de32791c-d8b8-4311-ae33-cd4b0e4501de.png">


# 2. Package categorization

## a. Scope

We divided our list of packages into 5 main categories : Plots, Tables, Stats, CDISC and Utilities. For the classification, I use the title and the description of the package. To clean the data, I use the library [Spacy](https://spacy.io/). The classification method is based on binary matching between the list of keywords for a category and the description/title of the package.


<img width="1523" alt="Package categorisation scope" src="https://user-images.githubusercontent.com/49449000/192283452-5af0498b-686f-4fcf-a7dc-5213c4944258.png">


## b. Performance measurement

We measure the performance using a test dataset containing 115 examples : 10 Plots, 8 Tables, 88 Stats, 2 CDISC and 15 Utilities (sum ≠ 115 bcz it's a multilabel classification). You have the accuracy on the following figure. **!!! As we have a strong imbalanced dataset, accuracy is not always relevant. To have better insights, you can calculate Precision, Recall and F1-score.**


<img width="1152" alt="Package categorisation - Performance" src="https://user-images.githubusercontent.com/49449000/192284783-f7c15d6e-2b89-4f18-bed4-4dcd279db018.png">



