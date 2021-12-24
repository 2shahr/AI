from sklearn.model_selection import train_test_split
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()


#################### Functions ############

def normalize(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min), min, max


def nave_bayese_model(X, Y):
    ClassNumber, ClassFrequency = np.unique(Y, return_counts=True)
    Ph_y = ClassFrequency/ClassFrequency.sum()
    temp_mu = []
    for classes in range(ClassNumber.shape[0]):
        temp = X[Y == classes].mean(axis=0)
        temp_mu.append(temp)

    Ph_X = np.array(temp_mu)
    return Ph_X, Ph_y, ClassNumber


def Prediction(X, ClassNumber, Ph_X, Ph_y):
    Classes = []
    for Data in range(X.shape[0]):
        Inp = X[Data]
        Probabilities = []
        for iter in range(ClassNumber.shape[0]):
            temp = Compute_probability(Inp, iter, Ph_X, Ph_y)
            Probabilities.append(temp)

        Probabilities = np.array(Probabilities)
        Class = ClassNumber[np.argmax(Probabilities)]
        Classes.append(Class)

    return np.array(Classes)


def Compute_probability(x, y, Ph_X, Ph_y):
    x = x.reshape(-1, 1)
    x = x.toarray()
    res = 1
    for j in range(x.shape[0]):
        Pxy = Ph_X[y][0, j]
        res *= (Pxy**x[j])*((1-Pxy)**(1-x[j]))
    return res * Ph_y[y]


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = re.sub(r'[\^]', '', words)
    new_words = re.sub(r'[.]', '', new_words)
    new_words = re.sub(r'[?]', '', new_words)
    new_words = re.sub(r'[!]', '', new_words)
    new_words = re.sub(r'[...]', '', new_words)
    new_words = re.sub(r'[#]', '', new_words)
    new_words = re.sub(r'[@]', '', new_words)
    new_words = re.sub(r'[,]', '', new_words)

    return new_words


def replace_not(words):
    """Replace all integer occurrences in list of tokenized words with textual representation"""
    new_words = re.sub(r"n\'t", " not", words)

    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    words = tt.tokenize(words)
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)

    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return s.str.replace(pattern, r"\1")


def preprocess(sample):
    sample = remove_URL(sample)
    words = remove_punctuation(sample)
    words = replace_not(words)
    words = remove_stopwords(words)
    temp = pd.DataFrame(words).values.tolist()

    Newwords1 = ' '
    for i in range(len(temp)):
        Newwords1 += ' ' + temp[i][0]

    return Newwords1[1:]


def get_frequence_features(data):
    vectorizer = CountVectorizer(
        analyzer='word', lowercase=True, stop_words='english',)
    features = vectorizer.fit_transform(data)  # Unigram features
    return features, vectorizer


#################### Main ############

# Load data
with open('imdb_labelled.txt') as f:
    lines = f.readlines()
Twittes = []
Labels = []
for i in range(len(lines)):
    Twittes.append(lines[i][0:-2])
    temp = lines[i][-2]
    if temp == '0':
        Labels.append(0)
    else:
        Labels.append(1)

Twittes = np.array(Twittes)
Labels = np.array(Labels)


# Pre-processing

PreProcessedTwittes = []
for i in range(Twittes.shape[0]):
    PreProcessedTwittes.append(preprocess(Twittes[i]))

PreProcessedTwittes = np.array(PreProcessedTwittes)


# Feature extraction
Features, vectorizer0 = get_frequence_features(PreProcessedTwittes)

#features0 = vectorizer0.transform(train_sentences_prelucrate_N2)


# Normalize
Inputs, minX, maxX = normalize(Features)


# Spliting
x_trn, x_tst, y_trn, y_tst = train_test_split(
    Inputs, Labels, test_size=0.2, random_state=42)
y_trn = y_trn.reshape(y_trn.shape[0])
y_tst = y_tst.reshape(y_tst.shape[0])


# Train
Ph_X, Ph_y, ClassNumber = nave_bayese_model(x_trn, y_trn)


# Test
Y_GLDA_Train = Prediction(x_trn, ClassNumber, Ph_X, Ph_y)
Y_GLDA_Test = Prediction(x_tst, ClassNumber, Ph_X, Ph_y)


# Evaluation
Accuracy_trn = (np.where(y_trn == Y_GLDA_Train)[0].shape[0]) / (y_trn.shape[0])
print('Train accuracy: ', Accuracy_trn)

Accuracy_tst = (np.where(y_tst == Y_GLDA_Test)[0].shape[0]) / (y_tst.shape[0])
print('Test accuracy: ', Accuracy_tst)


TP = np.where((y_trn == 0) & (Y_GLDA_Train == 0))[0].shape[0]
FP = np.where((y_trn == 1) & (Y_GLDA_Train == 0))[0].shape[0]
FN = np.where((y_trn == 0) & (Y_GLDA_Train == 1))[0].shape[0]
TN = np.where((y_trn == 1) & (Y_GLDA_Train == 1))[0].shape[0]
Accuracy_trn0 = TP / (TP+FN)
Accuracy_trn1 = TN / (TN+FP)
print('Train accuracy per class 0: ', Accuracy_trn0,
      'Train accuracy per class 1: ', Accuracy_trn1)

TP = np.where((y_tst == 0) & (Y_GLDA_Test == 0))[0].shape[0]
FP = np.where((y_tst == 1) & (Y_GLDA_Test == 0))[0].shape[0]
FN = np.where((y_tst == 0) & (Y_GLDA_Test == 1))[0].shape[0]
TN = np.where((y_tst == 1) & (Y_GLDA_Test == 1))[0].shape[0]
Accuracy_tst0 = TP / (TP+FN)
Accuracy_tst1 = TN / (TN+FP)
print('Train accuracy per class 0: ', Accuracy_tst0,
      'Train accuracy per class 1: ', Accuracy_tst1)
