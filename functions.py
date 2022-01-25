# Function for Expanding Contractions
import contractions
from contractions import contractions_dict
import re
import nltk
# Removing contractions   
def expand_contractions(text):
    contraction_mapping=contractions_dict
    words = ' '.join(text)
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]

        return expanded_contraction
    try:
        expanded_text = contractions_pattern.sub(expand_match, words)
        expanded_text = re.sub("'", "", expanded_text)
    except:
        return words
    return expanded_text


# remove symbols function
def rem_punct(words):
    res = re.sub(r'[^\w\s\']', '', str(words))
    return res

# remove numbers function
def rem_nums(words):
    rem_num = re.sub('\d+', ' ', str(words))
    return rem_num

# Converting lowercase function
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []

    for word in words:

        new_word = word.lower()
        new_words.append(new_word)
    return new_words

# Stopword function
from nltk.corpus import stopwords


def remove_stopwords(words):
    stopword_list=set(stopwords.words('english'))
    stopword_list.remove('no')
    stopword_list.remove('not')
    stopword_list.update(["href","product", "earphone", "realme"])
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

# Lemmatization function
import spacy
nlp = spacy.load('en_core_web_sm')
def lemmatize(sample):
    doc = nlp(sample)

    tokens = []
    for token in doc:
        tokens.append(token)

    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return lemmatized_sentence

# tokenization function
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# running all functions in sequence
def normalize_and_lemmaize(text):
    sample = rem_punct(text)
    tokenized = tokenize(sample)
    lower = to_lowercase(tokenized)
    stop = remove_stopwords(lower)
    expand = expand_contractions(stop)
    lemmatized = lemmatize(expand)
    rem_rum = rem_nums(lemmatized)
    return ''.join(rem_rum)
