
import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
wordDict = Counter()


def MakePreprocessing(Input):
    temp = re.compile('[^a-zA-Z]')
    Tokens = tt.tokenize(Input)
    new_Tokens = []
    for Token in Tokens:                
        Token = temp.sub('', Token)
        if Token not in stopwords.words('english') and len(Token)>2:
            new_Tokens.append(Token.lower())   
            
    temp = pd.DataFrame(new_Tokens).values.tolist()
    
    Output = ' '
    for i in range(len(temp)):
        Output+=' ' +temp[i][0]

    return Output[1:]

def RNN(X_trn, y_trn):

# building a linear stack of layers with the sequential model
    model = Sequential() 
    
# Embedding layer
    vocab_size = 100    
    model.add(layers.Embedding(vocab_size, 100, input_length=X_trn.shape[1]))

# RNN layer
    model.add(layers.SimpleRNN(200, activation='relu')) 
    
# flatten output of conv
    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
 
    
# Fully connected layer
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(50, activation='sigmoid'))
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_trn, y_trn, epochs = 150, verbose = False, batch_size = 16)
    return model

def Prediction(Net, X_trn, y_trn, X_tst):
    Net.evaluate(X_trn, y_trn, verbose = False)
    Outs = np.round(Net.predict(X_tst))    
    
    return Outs

#################### Main ############ 

# Load data


Raw_data = pd.read_csv('IMDB_Dataset.csv',encoding='latin-1')
Temp = pd.DataFrame(Raw_data).to_numpy()

# Remove Nans
Data = Temp[~pd.isnull(Temp).any(axis=1)]

Texts = Data[:,0]
Rate = Data[:,1]
Rate[Rate=='positive'] = 1
Rate[Rate=='negative'] = 0


# Preprocessing
ProcessedTexts = []
abbr_dict = []
for i in range(Texts.shape[0]):
    ProcessedTexts.append(MakePreprocessing(Texts[i]))


# Convert each text to number sequence
tokenizer = Tokenizer(num_words = 20000)
tokenizer.fit_on_texts(ProcessedTexts)
sequence = tokenizer.texts_to_sequences(ProcessedTexts)


#  Add pad to sequence ends
Maxlen = maxLength = max(len(x) for x in sequence)
Padded = pad_sequences(sequence, padding='post', maxlen=Maxlen)

# Convert float to int
Padded = Padded.astype(int)



# Splitting dataset to train, validation and test
X_trn, X_tst, y_trn, y_tst = train_test_split(Padded, Rate, test_size=0.2)
y_trn = to_categorical(y_trn)
y_tst = to_categorical(y_tst)


Net = RNN(X_trn, y_trn)

Outs = Prediction(Net, X_trn, y_trn, X_tst)

print("Accuracy: ", accuracy_score(y_tst, Outs))
