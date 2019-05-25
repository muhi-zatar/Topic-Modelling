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

