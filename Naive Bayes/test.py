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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
dictionary = {"a": 10, "b": 20}
print(max(dictionary, key=dictionary.get))
del dictionary["b"]
print(max(dictionary, key=dictionary.get))