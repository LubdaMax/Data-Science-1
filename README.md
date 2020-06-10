# Data-Science-1


## Quellen Textanalyse & Zusammenfassung

### textanalyse
Videoreihe zu Textanalyse:
* https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL
* https://cognitiveclass.ai/courses/systemt

### Zusammenfassen
* https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715
* https://pdfs.semanticscholar.org/7e30/d0c7aaaed7fa2d04fc8cc0fd3af8e24ca385.pdf
* https://www.sciencedirect.com/topics/computer-science/extractive-summarization
* https://blog.floydhub.com/gentle-introduction-to-text-summarization-in-machine-learning/ Den habe ich mal gebaut, Zusammenfassung ist nur so semi-toll.
* https://stackabuse.com/text-summarization-with-nltk-in-python/
* https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
* https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/
* https://shodhganga.inflibnet.ac.in/bitstream/10603/278911/15/15_chapter%205.pdf
* https://link.springer.com/content/pdf/10.1007%2F3-540-36127-8.pdf

### Wichtige Tools, Datenbanken, Libaries
* https://www.nltk.org/
* https://www.tensorflow.org/datasets/catalog/overview

## Datasets
* https://www.kaggle.com/pariza/bbc-news-summary?
* https://github.com/awesomedata/awesome-public-datasets#naturallanguage
* https://lionbridge.ai/datasets/the-best-25-datasets-for-natural-language-processing
* https://arxiv.org/pdf/1810.09305.pdf
_________________________________________________________

1 DEFINE GOAL / PLAN
* welche Art von Text wollen wir verarbeiten? Paper, Artikel, Fachartikel, ..? Sprache? Corpora? 
-> Zeitungsartikel, englisch, verschiedene Themen

2 FIND OPEN DATA FROM 2 SOURCES
* input single oder multiple documents?
* BBC, CNN, ggf. DUC

3 MERGE AND CLEAN DATA 


4 TEST AND VERIFY DATA QUALITY
* wie können wir Datenqulität testen?
* abstractedness (n-grams), compression ratio 

5 DATA PROCESSING (OPTIONAL)
* tokenizing, stemming, etc. (e.g. youtube tutorials sentdex)
* labelling von Sätzen

6 APPLY 2 DIFFERENT ALGOS
* steps: intermediate representation, scoring, selection of summary
* topic representation vs. indicator representation
* machine learning (classification problem), e.g. Hidden Markov, Conditional Random Fields // geeignetes Datenset? (labelled data)
* Methode aus package nutzen? (pytextrank)

7 EVALUATE AND VERIFY RESULTS
* pyrouge packages
* e.g. F1 score for Rouge-1/-2/-L, METEOR


8 CONCLUSION
test
