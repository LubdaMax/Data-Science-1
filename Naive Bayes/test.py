import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import re
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
def get_vectors(strs):
    text = strs
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

print(get_vectors(["This is a sentence", "Sentence is great"]))

vectors = get_vectors(["This is a sentence", "Sentence is great"])
centroid = np.zeros(len(vectors[0]))
for i in vectors:
    centroid += i
centroid = centroid/len(vectors)

for i in vectors:
    print(cosine_similarity([i, centroid]))

print(centroid)
