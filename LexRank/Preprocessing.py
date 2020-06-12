#import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import pandas as pd
import os
import nltk
from pathlib import Path
#import re
#import tensorflow_datasets as tfds


# Initialize the dataframe used to store Articles in col. 0
index = np.linspace(0, 509, 510)
total_data = pd.DataFrame(columns=["Article","Category"], index=index)
total_data = total_data.fillna(0)

# Relative path of dataset
rootpath = Path.cwd()
articlepath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\News Articles")
summarypath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\Summaries")


# Insert data (ALL articles) from .txt files into the dataframe
#categories = ["business", "entertainment","politics","sport","tech"]

# Insert data (only business articles) from .txt files into the dataframe
categories = ["business"]

count = 0
for category in categories:
    articlepath = articlepath_folder + "\\" + category
    for entry in os.scandir(articlepath):
        text_file = open(entry, "r")
        raw_text = text_file.read()
        total_data.iloc[count, 0] = raw_text
        total_data.iloc[count,1] = category
        text_file.close()

        count += 1

