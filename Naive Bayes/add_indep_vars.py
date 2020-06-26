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

# Filepath which reads a dictionary consisting of the Wikihow articles. Each article is part of a Pandas Dataframe which has 
# one row per sentence. The first column ("sentence") holds the sentences as strings. 
# The second column ("in_Summary")holds the dependent binary variable which is 1, if a sentence is in the provided summary, zero otherwise.
# This file adds all requiered independent variables for classification to the dataframes
# of the Wikihow dataset.
rootpath = Path.cwd()
filename = Path.joinpath(rootpath, r"Pre-Processing & EDA\wiki_data_processed_no_overview_notitle_10k")
infile = open(filename,'rb')
article_dict = pickle.load(infile)
infile.close()


def pos(Dataframe):
    """Adds the position within the Article as independent variable to the dataframe.
    """
    Dataframe["pos"] = np.linspace(1, Dataframe.shape[0], Dataframe.shape[0])

    return Dataframe


def Relative_pos(Dataframe):
    """Adds the normalized position within the Article as independent variable to the dataframe.
    """
    Dataframe["rel_pos"] = ""
    num_sentences = Dataframe.shape[0]

    for s in range(num_sentences):
        Dataframe.iloc[s, 3]= Dataframe.iloc[s, 2]/ num_sentences

    return Dataframe


def TF_ISF(Dataframe):
    """Adds the Term-Frequency Inverse-Sentence-Frequency as independent variable to the dataframe."""
    sentences = Dataframe["sentence"]
    sentences = sentences.to_frame()
    ps = nltk.stem.PorterStemmer()
    stop = set(stopwords.words("english"))
    tokenizer = nltk.RegexpTokenizer(r'\w+|\d+')
    num_sentences = sentences.shape[0]

    # Word tokenization, stopword removal, stemming
    for sentence in range(num_sentences): 
        sentences.iloc[sentence, 0] = tokenizer.tokenize(sentences.iloc[sentence, 0])
        
        for i in range(len(sentences.iloc[sentence, 0])):
            sentences.iloc[sentence, 0][i] = ps.stem(sentences.iloc[sentence, 0][i].lower())

        sentences.iloc[sentence, 0] = [w for w in sentences.iloc[sentence, 0] if not w in stop]
    #Calculate Term Frequency (within sentence) and Inverse Sentence Frequency
    sentences["TF"] = "" # Term Frequency
    sentences["SF"] = "" # Sentence Frequency
    sentences["ISF"] = "" # Inverse Sentence Frequency

    for i in range(num_sentences):
        sentences.iloc[i, 1] = []
        sentences.iloc[i, 2] = []
        sentences.iloc[i, 3] = []

        for w in range(len(sentences.iloc[i, 0])):
            TF = 0
            SF = 0

            for v in range(len(sentences.iloc[i, 0])):
                if sentences.iloc[i, 0][w] == sentences.iloc[i, 0][v]:
                    TF += 1
            sentences.iloc[i, 1].append(TF)

            for v in range(num_sentences):
                if sentences.iloc[i, 0][w] in sentences.iloc[v, 0]:
                    SF += 1
            sentences.iloc[i, 2].append(SF)

            ISF = np.log(num_sentences / SF)
            sentences.iloc[i, 3].append(ISF)
    
    # Calculate Averge TF-ISF
    sentences["Avg-TF-ISF"] = ""
    for s in range(num_sentences):
        Avg_TF_ISF = 0

        for w in range(len(sentences.iloc[s, 0])):
            Avg_TF_ISF += sentences.iloc[s, 1][w] * sentences.iloc[s, 3][w] / len(sentences.iloc[s, 0])

        sentences.iloc[s, 4] = Avg_TF_ISF

    sentences.columns = ["sentence", "TF", "SF", "ISF", "Avg-TF-ISF"]
    Dataframe["Avg-TF-ISF"] = sentences["Avg-TF-ISF"]

    return Dataframe


def rel_s_lenght(Dataframe):
    """Adds the normalized lenght of the sentence (in characters) as independent variable to the dataframe."""
    Dataframe["rel_len"] = ""
    num_sentences = Dataframe.shape[0]
    max_characters = 0

    for s in range(num_sentences):
        if len(Dataframe.iloc[s, 0]) > max_characters:
            max_characters = len(Dataframe.iloc[s, 0])
    
    for s in range(num_sentences):
        Dataframe.iloc[s, 5] = len(Dataframe.iloc[s, 0]) / max_characters

    return Dataframe


def centroid_similarity_s2s_cohesion(Dataframe):
    """Adds cohesion beteween sentences and similarity to the centroid as independent variables to the dataframe.
    The centroid is the average of the vectorized representations of the sentences.
    """
    def get_vectors(strs):
        """returns matrix of vectorized sentences"""
        text = strs
        vectorizer = CountVectorizer()
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()


    def get_cosine_sim(strs):
        """Returns matrix of cosine similarities""" 
        vectors = [t for t in get_vectors(strs)]
        return cosine_similarity(vectors)


    sentences = Dataframe["sentence"]
    sentences = sentences.to_frame()
    num_sentences = Dataframe.shape[0]
    ps = nltk.stem.PorterStemmer()
    stop = set(stopwords.words("english"))
    tokenizer = nltk.RegexpTokenizer(r'\w+|\d+')
    
    # Word tokenization, stopword removal, stemming, return sentences as string in a list to create matrices
    sentence_list = []
    for s in range(num_sentences): 
        sentences.iloc[s, 0] = tokenizer.tokenize(sentences.iloc[s, 0])
        
        for i in range(len(sentences.iloc[s, 0])):
            sentences.iloc[s, 0][i] = ps.stem(sentences.iloc[s, 0][i].lower())

        sentences.iloc[s, 0] = [w for w in sentences.iloc[s, 0] if not w in stop]
        sentence_string = ""

        for w in sentences.iloc[s, 0]:
            sentence_string += w
            sentence_string += " "

        sentence_string = sentence_string[:-1]
        sentence_string += "."
        sentence_list.append(sentence_string)
    
    # Create Cosine Matrix
    cosine_matrix = get_cosine_sim(sentence_list)
    
    # Calculate similarity between sentence s and all other sentences in the document.
    cohesions = np.array([])
    for s in range(num_sentences):
        cohesion = sum(cosine_matrix[:, s]) - 1
        cohesions = np.append(cohesions, cohesion)

    # Normalize value
    rel_cohesions = cohesions / cohesions.max()
    Dataframe["rel_s2s_cohs"] = rel_cohesions

    # Create the centroid
    vector_matrix = get_vectors(sentence_list)
    centroid = np.zeros(len(vector_matrix[0]))
    for i in vector_matrix:
        centroid += i
    centroid = centroid/len(vector_matrix)

    # create matrix with centroid ontop and calculate cosine similarities for sentences with centroid
    # Normalize the value
    centroid_matrix = np.vstack((centroid, vector_matrix))
    centroid_cosine_matirx = cosine_similarity(centroid_matrix)

    centroid_similarity = []
    for s in range(num_sentences):
        similarity = centroid_cosine_matirx[s+1, 0] / max(centroid_cosine_matirx[1:, 0])
        centroid_similarity.append(similarity)
    
    Dataframe["centroid_sim"] = centroid_similarity

    return Dataframe
        

def named_entity(Dataframe):
    """Detects whether a sentence has one or more named entitiy. Adds a binary independent 
    variable to the Dataframe.
    """
    Dataframe["named_ent"] = ""
    num_sentences = Dataframe.shape[0]
    sp = spacy.load('en_core_web_sm')

    for s in range(num_sentences):
        sentence = sp(Dataframe.iloc[s, 0])
        if len(sentence.ents) > 0:
            Dataframe.iloc[s, 8] = 1

        else:
            Dataframe.iloc[s, 8] = 0

    return Dataframe


def main_concept(Dataframe):
    """Detects whether a sentence has one of the 15 (or less for small articles)
    most frequent nouns in it. Adds binary independent variable to the Dataframe."""
    Dataframe["main_con"] = ""
    num_sentences = Dataframe.shape[0]
    sp = spacy.load('en_core_web_sm')
    nouns = {} # dict of all nouns
    main_concepts = [] # list of most frequent nouns
    noun_tags = ["NNP", "NN", "NNPS", "NNS"]
    
    # Counts frequency of all nouns
    for s in range(num_sentences):
        sentence = sp(Dataframe.iloc[s, 0])
        for token in sentence:
            if token.tag_ in noun_tags:
                if token.lemma_ not in nouns:
                    nouns[token.lemma_] = 1
                if token.lemma_ in nouns:
                    nouns[token.lemma_] += 1
            else:
                pass

    # Adds most frequent nouns to list
    for w in range(min(int(len(nouns)*0.3), 15)):
        top_word = max(nouns, key=nouns.get)
        main_concepts.append(top_word)
        del nouns[top_word]

    # Adds one, if most frequent noun in the sentence
    # zero otherwise.
    for s in range(num_sentences):
        sentence = sp(Dataframe.iloc[s, 0])
        for token in sentence:
            if token.lemma_ in main_concepts:
                Dataframe.iloc[s, 9] = 1
                break
            if token.lemma_ not in main_concepts:
                Dataframe.iloc[s, 9] = 0
    
    return Dataframe


def add_indep(Dict):
    """Adds the indep. variables to the dataframes in the dictrionary."""
    for i in range(len(Dict)):
        Dict["Article{0}".format(i)] = pos(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = Relative_pos(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = TF_ISF(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = rel_s_lenght(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = centroid_similarity_s2s_cohesion(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = named_entity(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = main_concept(Dict["Article{0}".format(i)])
        print(i)
    return(Dict)


def pickle_save(Dict):
    """Saves the file in pikle format"""
    rootpath = Path.cwd()
    outfile = open(Path.joinpath(rootpath, r"Naive Bayes\wiki_data_indep_8_no_overview_notilte_10k"), 'wb')
    pickle.dump(Dict, outfile)
    outfile.close()


# test cases
test = pos(article_dict["Article3"])
test = Relative_pos(article_dict["Article3"])
test = TF_ISF(article_dict["Article3"])
test = rel_s_lenght(article_dict["Article3"])
test = centroid_similarity_s2s_cohesion(article_dict["Article3"])
test = named_entity(test)
test = main_concept(test)
print(test)


# Executes the addition of indep. variables for the whole dictionary
add_indep(article_dict)
pickle_save(article_dict)

# opens the dictionary with independent variables as testcase
rootpath = Path.cwd()
testopen = open(Path.joinpath(rootpath, r"Naive Bayes\wiki_data_indep_8_no_overview_notilte_10k"), 'rb')
idep_dict = pickle.load(testopen)
print(idep_dict["Article200"])
