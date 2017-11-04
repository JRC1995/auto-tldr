
## Extractive Summarization based on keywords extracted using RAKE

This is a simple implementation of extractive summarization based on the keywords extracted by RAKE -[Rapid Automatic Keyword Extraction](https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents)

The implementation is inspired from http://smmry.com/about (which I recently discovered through [auto-tl;dr Reddit bot](https://www.reddit.com/user/autotldr) - which too I discovered recently at r/machine-learning). 

The high level description of the algorithm in http://smmry.com/about seemed pretty simple, and I wanted to try implementing it myself. After recently implemented RAKE recently, I already had codes for keyword extraction. With keyword extraction being settled, generating simplistic keyword-based extractive summarization is only a bit more work. 

Here's a sample summarization executed by this implementation:

### ORIGINAL TEXT: 
[(Source)](https://www.reddit.com/r/autotldr/comments/31b9fm/faq_autotldr_bot/)

Autotldr is a bot that uses SMMRY to create a TL;DR/summary. I will put forth points that address the effects this bot has on the reddit community.

It doesn't create laziness, it only responds to it

For the users who click the article link first and then return back to the comments, they will have already given their best attempt of fully reading the article. If they read it fully, the tl;dr is unneeded and ignored. If they skimmed or skipped it, the bot will be useful to at least provide more context to the discussion, like an extension of the title. A large portion of users, especially in the defaulted mainstream subreddits like /r/politics, don't even go to the article and go straight to the comments section. Most of the time, if I skip to the comments, I'm able to illicit some sort of understanding of what the article was about from the title and discussion. However this bot is able to further improve my conjectured understanding. It did not make me skip it, it only helped me when I already decided to skip it. The scenario in which this bot would create a significantly lazy atmosphere is if the tl;dr were to be presented parallel to the main submission, in the same way the OP's tl;dr is presented right next to the long body of self post. Also, the tl;dr becomes more prevalent/hidden as it will get upvoted/downvoted depending on how much of a demand there was for a tl;dr in the first place. If it becomes the top voted comment than it has become more of a competitor to the original text for those who go to the comments first, but by then the thread has decided that a tl;dr was useful and the bot delivered.

It can make sophisticated topics more relevant to mainstream Reddit

Sophisticated and important topics are usually accompanied or presented by long detailed articles. By making these articles and topics relevant to a larger portion of the Reddit userbase (those who weren't willing to read the full article), it popularizes the topic and increases user participation. These posts will get more attention in the form of upvotes/downvotes, comments, and reposts. This will increase the prevalence of sophisticated topics in the mainstream subreddits and compete against cliched memes. This has the potential of re-sophisticating the topic discussion in the mainstream subreddits, as more hardcore redditors don't have to retreat to a safe haven like /r/TrueReddit. This is a loose approximation and the magnitude of this effect is questionable, but I'm not surprised if the general direction of the theory is correct. I'm not claiming this would improve reddit overnight, but instead very very gradually.

It decreases Reddit's dependency on external sites

The bot doubles as a context provider for when a submission link goes down, is removed, or inaccessible at work/school. The next time the article you clicked gives you a 404 error, you won't have to depend on the users to provide context as the bot will have been able to provide that service at a much faster and consistent rate than a person. Additionally, an extended summary is posted in /r/autotldr, which acts as a perpetual archive and decreases how much reddit gets broken by external sites.

Only useful tl;dr's are posted

There are several criteria for a bot to post a tl;dr. It posts the three most important sentences as decided by the core algorithm, and they must be within 450-700 characters total. The final tl;dr must also be 70% smaller than the original, that way there is a big gap between the original and the tl;dr, hence only very long articles get posted on. This way the likelihood of someone nonchalantly declaring "TL;DR" in a thread and the bot posting in the same one is high. Also my strategy is to tell the bot to post in default, mainstream subreddits were the demand for a TL;DR is much higher than /r/TrueReddit and /r/worldevents.

Feel free to respond to these concepts and to raise your own. Be polite, respectful, and clarify what you say. Any offending posts to this rule will be removed.


### GENERATED SUMMARY:

Autotldr is a bot that uses SMMRY to create a TL ; DR/summary.

It can make sophisticated topics more relevant to mainstream Reddit Sophisticated and important topics are usually accompanied or presented by long detailed articles.

By making these articles and topics relevant to a larger portion of the Reddit userbase ( those who were n't willing to read the full article ) , it popularizes the topic and increases user participation.

This has the potential of re-sophisticating the topic discussion in the mainstream subreddits , as more hardcore redditors do n't have to retreat to a safe haven like /r/TrueReddit.

It decreases Reddit 's dependency on external sites The bot doubles as a context provider for when a submission link goes down , is removed , or inaccessible at work/school.

Additionally , an extended summary is posted in /r/autotldr , which acts as a perpetual archive and decreases how much reddit gets broken by external sites.

The final tl ; dr must also be 70 % smaller than the original , that way there is a big gap between the original and the tl ; dr , hence only very long articles get posted on.



```python
filename = 'summarytest.txt' #Enter Filename
```

### Loading the data into Python variable from the file. 


```python
file = open(filename,'r')
Text = ""
for line in file.readlines():
    Text+=str(line)
    Text+=" "
file.close()
```

### Removing non-printable characters from data and tokenizing the resultant test.


```python
import nltk
from nltk import word_tokenize
import string

def clean(text):
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text) #filter funny characters, if any.
    return text

Cleaned_text = clean(Text)

text = word_tokenize(Cleaned_text)
case_insensitive_text = word_tokenize(Cleaned_text.lower())
```

### Sentence Segmentation

Senteces is a list of segmented sentences in the natural form with case sensitivity. This will be
necessary for displaying the summary later on.

Tokenized_sentences is a list of tokenized segmented sentences without case sensitivity. This will be
useful for text processing later on. 


```python
# Sentence Segmentation

sentences = []
tokenized_sentences = []
sentence = " "
for word in text:
    if word != '.':
        sentence+=str(word)+" "
    else:
        sentences.append(sentence.strip())
        tokenized_sentences.append(word_tokenize(sentence.lower().strip()))
        sentence = " "
```

### Lemmatization of tokenized words. 


```python
from nltk.stem import WordNetLemmatizer

def lemmatize(POS_tagged_text):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    adjective_tags = ['JJ','JJR','JJS']
    lemmatized_text = []
    
    for word in POS_tagged_text:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
    
    return lemmatized_text

#Pre_processing:

POS_tagged_text = nltk.pos_tag(case_insensitive_text)
lemmatized_text = lemmatize(POS_tagged_text)
```

### POS Tagging lemmatized words

This will be useful for generating stopwords. 


```python
Processed_text = nltk.pos_tag(lemmatized_text)
```

### Stopwords Generation

Based on the assumption that typically only nouns and adjectives are qualified as parts of keyword phrases, I will include any word that aren't tagged as a noun or adjective to the list of stopwords. (Note: Gerunds can often be important keywords or components of it. But including words tagged as 'VBG' (tag for present participles and gerunds) also include verbs of present continiuous forms which should be treated as stopwords. So I am not adding 'VBG' to list of POS that should not be treated as 'stopword-POSs'. Punctuations will be added to the same list (of stopwords). Additional the long list of stopwords from https://www.ranks.nl/stopwords are also added to the list. 


```python
def generate_stopwords(POS_tagged_text):
    stopwords = []
    
    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','FW'] #may be add VBG too
    
    for word in POS_tagged_text:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])
            
    punctuations = list(str(string.punctuation))
    stopwords = stopwords + punctuations
    
    stopword_file = open("long_stopwords.txt", "r")
    #Source = https://www.ranks.nl/stopwords

    for line in stopword_file.readlines():
        stopwords.append(str(line.strip()))

    return set(stopwords)

stopwords = generate_stopwords(Processed_text)
```

### Partioning text into tokenized phrases using stopwords as delimeters.

The result can be somewhat like n-gram parsing.
All these partitioned phrases will serve as <b>candidate keywords</b>.


```python
def partition_phrases(text,delimeters):
    phrases = []
    phrase = " "
    for word in text:
        if word in delimeters:
            if phrase!= " ":
                phrases.append(str(phrase).split())
            phrase = " "
        elif word not in delimeters:
            phrase+=str(word)
            phrase+=" "
    return phrases

phrase_list = partition_phrases(lemmatized_text,stopwords)
```

### Partioning each segmented tokenizedsentences into tokenized phrases using stopwords as delimeters.

The tokenized segmented sentences are lemmatized before partioning.



```python

phrase_partitioned_sentences = []

for sentence in tokenized_sentences:
    POS_tagged_sentence = nltk.pos_tag(sentence)
    lemmatized_sentence = lemmatize(POS_tagged_sentence)
    phrase_partitioned_sentence = partition_phrases(lemmatized_sentence,stopwords)
    phrase_partitioned_sentences.append(phrase_partitioned_sentence)
```

### Scoring each words in the sum total of phrases using RAKE Algorithm

See more on RAKE here: https://github.com/JRC1995/RAKE-Keyword-Extraction


```python
# keyword scoring

from collections import defaultdict
from __future__ import division

frequency = defaultdict(int)
degree = defaultdict(int)
word_score = defaultdict(float)

vocabulary = []

for phrase in phrase_list:
    for word in phrase:
        frequency[word]+=1
        degree[word]+=len(phrase)
        if word not in vocabulary:
            vocabulary.append(word)
            
for word in vocabulary:
    word_score[word] = degree[word]/frequency[word]
```

### Scoring each phrase (candidate keyword) using RAKE

Furthermore the tokenized phrases are converted into a presentable format and put into the list named 
'keywords'


```python
import numpy as np

index=0
phrase_scores = np.zeros((len(phrase_list)),dtype=np.float32)
keywords = []

for phrase in phrase_list:
    for word in phrase:
        phrase_scores[index] += word_score[word]
    index+=1

for i in xrange(0,len(phrase_list)):
    
    keyword=''
    for word in phrase_list[i]:
        keyword += str(word)+" "
    
    keyword = keyword.strip()
    keywords.append(keyword)
```

### Ranking Keywords

Keywords are ranked based on their assigned scores.
The top keywords_num no. of highest scoring keywords are considered important; rest are ignored.
Sorted_keywords includes the sorted list of the top keywords_num no. of highest scoring keywords.
Tokenized_keywords contain the same keywords but in a tokenized mannner.



```python
sorted_index = np.flip(np.argsort(phrase_scores),0)

tokenized_keywords = []
sorted_keywords = []

keywords_num = 0
threshold = 50
if len(keywords)<threshold:
    keywords_num = len(keywords)
else:
    keywords_num = threshold

for i in xrange(0,keywords_num):
    sorted_keywords.append(keywords[sorted_index[i]])
    tokenized_keywords.append(sorted_keywords[i].split())

```

### Scoring Sentences

Sentences are scored by adding the scores of all the keywords that are present in the sentence, 
and also in the tokenized_keywords list (which contains only the relatively important keywords).


```python

sentence_scores = np.zeros((len(sentences)),np.float32)
i=0
for sentence in phrase_partitioned_sentences:
    for phrase in sentence:
        if phrase in tokenized_keywords:
            
            matched_tokenized_keyword_index = tokenized_keywords.index(phrase)
            
            corresponding_sorted_keyword = sorted_keywords[matched_tokenized_keyword_index]
            
            keyword_index_where_the_sorted_keyword_is_present = keywords.index(corresponding_sorted_keyword)
            
            sentence_scores[i]+=phrase_scores[keyword_index_where_the_sorted_keyword_is_present]
    i+=1

```

### Summary Generation

Given some hyperparameters the program computes the summary_size.
Sentences are then ranked in accordance to their corresponding scores. More precisely, the indices of the sentences are sorted based on the scores of their corresponding sentences. Based on size of the summary, indices of top 'summary_size' no. of highest scoring input sentences are chosen for generating the summary.
Summary is then generated by presenting the sentences (whose indices were chosen) in a chronological order.

Note: I hardcoded the selection of the first statement (if the summary_size is computed to be more than 1) 
because the first sentence can usually serve as an introduction, and provide some context to the topic. 


```python
Reduce_to_percent = 20
summary_size = int(((Reduce_to_percent)/100)*len(sentences))

if summary_size == 0:
    summary_size = 1

sorted_sentence_score_indices = np.flip(np.argsort(sentence_scores),0)

indices_for_summary_results = sorted_sentence_score_indices[0:summary_size]

summary = "\n"

current_size = 0

if 0 not in indices_for_summary_results and summary_size!=1:
    summary+=sentences[0]
    summary+=".\n\n"
    #current_size+=1
else:
    summary_size+=1


for i in xrange(0,len(sentences)):
    if i in indices_for_summary_results:
        summary+=sentences[i]
        summary+=".\n\n"
        current_size += 1
    if current_size == summary_size:
        break

print "\nSUMMARY: "
print summary
```

    
    SUMMARY: 
    
    Autotldr is a bot that uses SMMRY to create a TL ; DR/summary.
    
    It can make sophisticated topics more relevant to mainstream Reddit Sophisticated and important topics are usually accompanied or presented by long detailed articles.
    
    By making these articles and topics relevant to a larger portion of the Reddit userbase ( those who were n't willing to read the full article ) , it popularizes the topic and increases user participation.
    
    This has the potential of re-sophisticating the topic discussion in the mainstream subreddits , as more hardcore redditors do n't have to retreat to a safe haven like /r/TrueReddit.
    
    It decreases Reddit 's dependency on external sites The bot doubles as a context provider for when a submission link goes down , is removed , or inaccessible at work/school.
    
    Additionally , an extended summary is posted in /r/autotldr , which acts as a perpetual archive and decreases how much reddit gets broken by external sites.
    
    The final tl ; dr must also be 70 % smaller than the original , that way there is a big gap between the original and the tl ; dr , hence only very long articles get posted on.
    
    



```python


```
