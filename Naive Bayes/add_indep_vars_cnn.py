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

rootpath = Path.cwd()
filename = Path.joinpath(rootpath, r"cnn_articles_dict")
infile = open(filename,'rb')
article_dict = pickle.load(infile)
infile.close()
#print(article_dict["Article1"])

for i in range(len(article_dict)):
    article_dict["Article{0}".format(i)].columns = ["sentence"]

def pos(Dataframe):
    Dataframe["pos"] = np.linspace(1, Dataframe.shape[0], Dataframe.shape[0])

    return Dataframe


def Relative_pos(Dataframe):
    Dataframe["rel_pos"] = ""
    num_sentences = Dataframe.shape[0]

    for s in range(num_sentences):
        Dataframe.iloc[s, 2]= Dataframe.iloc[s, 1]/ num_sentences

    return Dataframe


def TF_ISF(Dataframe):
    sentences = Dataframe["sentence"]
    sentences = sentences.to_frame()
    ps = nltk.stem.PorterStemmer()
    stop = set(stopwords.words("english"))
    tokenizer = nltk.RegexpTokenizer(r'\w+|\d+')
    num_sentences = sentences.shape[0]

    # Word tokenization, stopword removal, stemming
    for sentence in range(num_sentences): 
        #sentences.iloc[sentence, 0] = sentences.iloc[sentence, 0].translate(string.punctuation)
        #sentences.iloc[sentence, 0] = nltk.word_tokenize(sentences.iloc[sentence, 0])
        sentences.iloc[sentence, 0] = tokenizer.tokenize(sentences.iloc[sentence, 0])
        
        for i in range(len(sentences.iloc[sentence, 0])):
            sentences.iloc[sentence, 0][i] = ps.stem(sentences.iloc[sentence, 0][i].lower())

        sentences.iloc[sentence, 0] = [w for w in sentences.iloc[sentence, 0] if not w in stop]
    #Calculate Term Frequency (within sentence) and Inverse Sentence Frequency
    sentences["TF"] = ""
    sentences["SF"] = ""
    sentences["ISF"] = ""

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
    #Dataframe["Sentence"] = sentences["Sentence"]
    #Dataframe["TF"] = sentences["TF"]
    #Dataframe["SF"] = sentences["SF"]
    #Dataframe["ISF"] = sentences["ISF"]
    Dataframe["Avg-TF-ISF"] = sentences["Avg-TF-ISF"]

    return Dataframe


def rel_s_lenght(Dataframe): 
    Dataframe["rel_len"] = ""
    num_sentences = Dataframe.shape[0]
    max_words = 0

    for s in range(num_sentences):
        if len(Dataframe.iloc[s, 0]) > max_words:
            max_words = len(Dataframe.iloc[s, 0])
    
    for s in range(num_sentences):
        Dataframe.iloc[s, 4] = len(Dataframe.iloc[s, 0]) / max_words

    return Dataframe


def title_similarity_s2s_cohesion(Dataframe):
    def get_vectors(strs):
        text = strs
        vectorizer = CountVectorizer()
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()


    def get_cosine_sim(strs): 
        vectors = [t for t in get_vectors(strs)]
        return cosine_similarity(vectors)


    sentences = Dataframe["sentence"]
    sentences = sentences.to_frame()
    num_sentences = Dataframe.shape[0]
    ps = nltk.stem.PorterStemmer()
    stop = set(stopwords.words("english"))
    tokenizer = nltk.RegexpTokenizer(r'\w+|\d+')
    sentence_list = []
    # Word tokenization, stopword removal, stemming, return to string
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
    
    matrix = get_cosine_sim(sentence_list)
    Dataframe["title_sim"] = np.zeros(num_sentences)

    cohesions = np.array([])
    for s in range(num_sentences):
        cohesion = sum(matrix[:, s]) - 1
        cohesions = np.append(cohesions, cohesion)

    rel_cohesions = cohesions / cohesions.max()
    Dataframe["rel_s2s_cohs"] = rel_cohesions
    # Drop title from Frame
    Dataframe = Dataframe.drop(index=[0])
    return Dataframe
        

def named_entity(Dataframe):
    Dataframe["named_ent"] = ""
    num_sentences = Dataframe.shape[0]
    sp = spacy.load('en_core_web_sm')

    for s in range(num_sentences):
        sentence = sp(Dataframe.iloc[s, 0])
        if len(sentence.ents) > 0:
            Dataframe.iloc[s, 7] = 1

        else:
            Dataframe.iloc[s, 7] = 0

    return Dataframe


def main_concept(Dataframe):
    Dataframe["main_con"] = ""
    num_sentences = Dataframe.shape[0]
    sp = spacy.load('en_core_web_sm')
    nouns = {}
    main_concepts = []
    noun_tags = ["NNP", "NN", "NNPS", "NNS"]
    
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
    
    for w in range(min(int(len(nouns)*0.3), 15)):
        top_word = max(nouns, key=nouns.get)
        main_concepts.append(top_word)
        del nouns[top_word]

    for s in range(num_sentences):
        sentence = sp(Dataframe.iloc[s, 0])
        for token in sentence:
            if token.lemma_ in main_concepts:
                Dataframe.iloc[s, 8] = 1
                break
            if token.lemma_ not in main_concepts:
                Dataframe.iloc[s, 8] = 0
    
    return Dataframe
    


def add_indep(Dict):
    for i in range(len(Dict)):
        Dict["Article{0}".format(i)] = pos(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = Relative_pos(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = TF_ISF(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = rel_s_lenght(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = title_similarity_s2s_cohesion(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = named_entity(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = main_concept(Dict["Article{0}".format(i)])
        print(i)
    return(Dict)


def pickle_save(Dict):
    rootpath = Path.cwd()
    outfile = open(Path.joinpath(rootpath, r"cnn_indep_8"), 'wb')
    pickle.dump(Dict, outfile)
    outfile.close()


"""
test = pos(article_dict["Article0"])
test = Relative_pos(article_dict["Article0"])
test = TF_ISF(article_dict["Article0"])
test = rel_s_lenght(article_dict["Article0"])
print(test)
"""


add_indep(article_dict)
pickle_save(article_dict)


rootpath = Path.cwd()
testopen = open(Path.joinpath(rootpath, r"cnn_indep_8"), 'rb')
idep_dict = pickle.load(testopen)
print(idep_dict["Article200"])
