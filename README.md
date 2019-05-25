# Topic-Modelling
In this repository, the data preprocessing stage, applying LDA and creating word embeddings word2vec are presented and detailed. 

# Data Preprocessing

After retrieving the Data from the database, it is mandatory to preprocess the data in order to apply Machine Learning techniques on it.

The preprocessing in our case is divided into multiple steps as follwoing:

1- Preprocessing using regular expressions.

2- Tokenization.

3- Preparing and removing stop words.

4- Create bag of words(bigram models).

5- Lemmatizing and stemming.

**Preprocessing using regular expressions**

[Regular expressions](https://www.machinelearningplus.com/python/python-regex-tutorial-examples/) is a very useful tool for preprocessing data such as; removing undesirable strings, spaces, etc. Which we will be using here for the purpose of text preprocessing.
```python
import re

def prepare_data(data): 
  data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
  data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
  data = [re.sub("\'", "", sent) for sent in data]

  return data
```
**Tokeinization**

Simply in this step, we're splitting sentences into words for purposes that will be more obvious later in the document. For doing this, [simple_preprocess from gensim](https://radimrehurek.com/gensim/utils.html) does the job in addition to removing punctuation and undesirable characters.
```python
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
```
**Preparing and removing stop words**

This step gives more emphasis on the words that are more relevant and help the learning technique to concentrate on them. Examples on stop words in english could be: 'and', 'but', 'a', 'how', 'what'. Words like these could occur in any text and hence it is better to remove them.

Stop words removal start with stating these words, luckily, we have them ready thank to [Natural Language Toolkit (nltk)](https://www.nltk.org/). 
The following lines of code prepares the stop words in english. 
```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# we can also extend our stopwords
stop_words.extend(['hello', '.com'])
def remove_stopwords(texts):
  stop_words = stopwords.words('english')
  # we can also extend our stopwords
  stop_words.extend(['hello', '.com'])
  return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
```

**Create bag of words (bigram models)**

Bigrams are two words frequently occurring together in the document. Again, they can be [created in gensim](https://radimrehurek.com/gensim/models/phrases.html)
```python
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)
def make_bigrams(texts, data_words):
  bigram = gensim.models.Phrases(data_words, min_count=1, threshold=10) # higher threshold fewer phrases.
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  return [bigram_mod[doc] for doc in texts]
```

**Lemmatizing and stemming**

Lemmatizing is changing past and future tenses to present tense and third point of view are changed to first point of view, whereas Stemming is simply convierting the word back to its root. Again these techniques help to unify the appearance of words that existed in different forms, as an example; rockets is converted back to rocket, walks, walked and walking are converted to walk. This helps the learning technique not to get confused by these form of the same word (after all, the machines are not as smart as us, so far!).
However, this is not as tiring as it sounds. It can be done using [WordNetlemmatizer](https://www.geeksforgeeks.org/python-lemmatization-with-nltk/) from nltk or lemmatizer from [spacy](https://spacy.io/api/lemmatizer), we will be using sapcy as it supports simple part of speech tagging(identifies if the word is verb, noun, adj, etc. 
```python
import spacy
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
  texts_out = []
  # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
  nlp = spacy.load('en', disable=['parser', 'ner'])
  for sent in texts:
      doc = nlp(" ".join(sent)) 
      texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
  return texts_out
```
# Apply Latent Dirichlet Allocation (LDA)

Latent Dirichlet Allocation (LDA) is an unsupervised learning algorithm for topic modeling. To tell briefly, LDA imagines a fixed set of topics. Each topic represents a set of words. And the goal of LDA is to map all the documents to the topics in a way, such that the words in each document are mostly captured by those imaginary topics. For more details, read the [paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), or [this article](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158) and have a look on [this video](https://www.youtube.com/watch?v=3mHy4OSyRf0). With Gensim, life is much easier for building this algorithm; you only have to predetermine the number of topics, get the data, clean it and gensim does the magic. 

But first of all, as known, machines do not understand words, and hence we need to represent each word by a differen id in a dictionary, and calculating the frequency of each term. This can be done using [doc2bow from gensim](https://radimrehurek.com/gensim/corpora/dictionary.html)
```python
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
```
Now we have everything ready to build the [LDA model](https://radimrehurek.com/gensim/models/ldamodel.html)
```python
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
```
# Interpreting and visualizing the results

Now we cant get the keywords in each topic by writing the following line:
```python
print(lda_model.print_topics())
```
The output of this line will be in the following format:
```
[(0,
  '0.046*"video" + 0.020*"right" + 0.017*"also" + 0.016*"watch" + 0.013*"have" '
  '+ 0.013*"congress" + 0.012*"first" + 0.009*"law" + 0.009*"interview" + '
  '0.008*"medium"'),
 (1,
  '0.028*"work" + 0.022*"could" + 0.015*"tell" + 0.013*"hear" + 0.013*"hate" + '
  '0.012*"build" + 0.011*"money" + 0.010*"mind" + 0.009*"modiji_modiji" + '
  '0.009*"next"'),
 (2,
  '0.040*"not" + 0.025*"do" + 0.021*"be" + 0.013*"great" + 0.010*"way" + '
  '0.010*"politic" + 0.010*"party" + 0.010*"political" + 0.008*"can" + '
  '0.007*"government"'),
 (3,
  '0.027*"really" + 0.024*"guy" + 0.020*"well" + 0.019*"bjp" + 0.017*"hai" + '
  '0.017*"modi" + 0.013*"long" + 0.013*"still" + 0.010*"someone" + 0.009*"ki"'),
 (4,
  '0.021*"people" + 0.020*"good" + 0.017*"get" + 0.014*"s" + 0.014*"make" + '
  '0.014*"say" + 0.013*"would" + 0.013*"trump" + 0.013*"think" + 0.012*"go"')]
```
How to interpret this?

Topic 0 is a represented as (0,
  '0.046*"video" + 0.020*"right" + 0.017*"also" + 0.016*"watch" + 0.013*"have" '
  '+ 0.013*"congress" + 0.012*"first" + 0.009*"law" + 0.009*"interview" + '
  '0.008*"medium"'),.

It means the top 10 keywords that contribute to this topic are: ‘video’, ‘right’, ‘also’.. and so on and the weight of ‘video’ on topic 0 is 0.046.

The weights reflect how important a keyword is to that topic.

**Data Visualization**
 
 There are multiple ways that we can visualize and represent our results from topic modeling using LDA. However, one popular way to do that is using [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html).
 ```
 import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
```
**Image of the results will be posted once ready**

