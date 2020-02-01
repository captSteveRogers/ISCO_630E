import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset with proper encoding :
dataset = pd.read_csv('spam_ham_data_set.csv', encoding='latin-1')
dataset = dataset[["v1","v2"]]
dataset = dataset.rename(columns={"v1":"label", "v2":"text"})
dataset.head()

print(dataset.info())
print(dataset.label.value_counts())

dataset["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

topMsg = dataset.groupby("text")["label"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)
print(topMsg)
dataset.drop_duplicates(keep=False, inplace=True)
print(dataset.shape)

import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")    #for crude estimation to root words

def cleanText(msg):
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in msg.split() if word.lower() not in stopwords.words("english")]
    return " ".join(words)

dataset["text"] = dataset["text"].apply(cleanText)
dataset = dataset[["label","text"]]
dataset.head()

X = dataset.iloc[:,1].values
y = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
text_train, text_test, label_train, label_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


messages = dataset.text.values
words_all = []
for message in messages:
    words_all += (message.split(" "))
unique_words = set(words_all)
dictionary_words = {i:words_all.count(i) for i in unique_words}
dictionary_words['hello']

spam_messages = dataset.text.values[dataset.label == "spam"]
spam_words = []
for spam in spam_messages:
    spam_words += (spam.split(" "))
unique_spam_words = set(spam_words)
dictionary_spam = {i:spam_words.count(i) for i in unique_spam_words}
dictionary_spam['win']

ham_messages = dataset.text.values[dataset.label == "ham"]
ham_words = []
for ham in ham_messages:
    ham_words += (ham.split(" "))
unique_ham_words = set(ham_words)
dictionary_ham = {i:ham_words.count(i) for i in unique_ham_words}
dictionary_ham['love']

total_words = len(words_all)
total_spam = len(spam_words)
total_ham = len(ham_words)
print(total_words, total_spam, total_ham)

def prob_spam(word):
    return (dictionary_spam[word]/total_spam) 

def prob_ham(word):
    return dictionary_ham[word]/total_ham 

def prob_word(word):
    try:
        return dictionary_words[word]/total_words
    except KeyError:
        return 0.000000001 

def prob_msg_spam(message):
    num = den = 1
    for word in message.split():
        if word in spam_words:
            num *= prob_spam(word)
            den *= prob_word(word)
    
    # Laplace Smoothing step 
    if den==0:
        num+=1
        den+=1
    return num/den

def prob_msg_ham(message): 
    num = den = 1
    for word in message.split():
        if word in ham_words:
            num *= prob_ham(word)
            den *= prob_word(word)
    # Laplace Smoothing step
    if den==0:
        num+=1
        den+=1
    return num/den

def spam_pred(msg):
    if prob_msg_spam(msg) >= prob_msg_ham(msg):
        return "spam"
    else:
        return "ham"
    
def accuracy(text_test, label_test):
    false_positive = false_negative = 0 
    true_positive = true_negative = 0
    for i,m in enumerate(text_test):
        predicted = spam_pred(m)
        actual = label_test[i]
        if predicted == "spam" and actual == "spam":
            true_negative+=1
        if predicted == "spam" and actual == "ham":
            false_negative+=1
        if predicted == "ham" and actual == "spam":
            false_positive+=1
        if predicted == "ham" and actual == "ham":
            true_positive+=1
    acc = (true_negative+true_positive)/len(text_test)
    return acc, false_positive, false_negative, true_positive, true_negative

acc,fp,fn,tp,tn = accuracy(text_test,label_test)
print(acc*100, fp,fn,tp,tn) 

print("True positive: ",tp,"\n False positive: ",fp,"\n False Negative: ",fn,"\n True Negative: ",tn)   