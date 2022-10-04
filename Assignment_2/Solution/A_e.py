import re
import json
import pandas as pd
import numpy as np
import sys
from collections import Counter
import time
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

snow_stemmer = SnowballStemmer(language='english', ignore_stopwords=True)

remove_word_puncts = re.compile(r"([0-9]|@|\#|\$|\%|\^|\\)")
remove_only_punct = re.compile(r"(\\x03|\\x16|\&quot|&amp|\&|\;|\(|\)|\*|\'|\+|,|\-|\!|\-|\/|\:|\;|=|>|\?|\_|`|\||\~|\{|\}|\[|\]|\")")

words_to_remove = ['i','the', 'and', 'of', 'to', 'is', 'it', 'its','this', 'that', 's','on', 'in', 'with', 'you', 'but', 'for', 'as','are', 'was', 'be', 'have', 'all', 'his','from', 'my', 'he', 'has', 'do', 'they', 'her', 'me', 'an', 'so','if', 'by']

def get_stem(word):
    prev = ""
    while(prev != word):
        prev = word
        word = snow_stemmer.stem(word)
    return word

def normalise_word(word):
    temp = ""
    for idx,char in enumerate(word):
        if(idx < 2):
            temp += char
        else:
            if(not (char.lower() == word[idx-1].lower() and char.lower() == word[idx-2].lower())):
                temp += char
    return temp

def preprocess(review):
    review = review.replace("n`t"," not")
    review = review.replace("n't"," not")
    
    sentences = []
    for sent in review.split("."):
        words = []
        for word in sent.split(" "):
            word = word.lower()
            if(remove_word_puncts.search(word)):
                continue
            else:
                s = remove_only_punct.sub(" ",word).strip()
                s = re.sub(r"[^a-zA-Z\s\.]"," ",s)
                s = re.sub("/s/s*"," ",s).strip()
                if(len(s) > 1 and s != "" and (not s in words_to_remove)):
                    words.append(s)
        
        sentence = " ".join(words).strip()
        if(len(sentence) > 1):
            sentences.append(sentence+".")
    
    return ".".join(sentences).strip().lower()

def compute_log_posterior(word, label):
    posterior = round(np.log((class_vocab[label].get(word,0)+1)/(class_word_count[label]+len(all_vocab))),6)
    return posterior

stemming_cache = {}
def predict_class_bigram(review):
    processed_review = preprocess(review)
    bigrams = []
    for sent in processed_review.split("."):
        words = sent.split(" ")
        for idx in range(len(words)-1): 
            w1 = words[idx].strip().lower()
            stemmed_w1 = stemming_cache.get(w1,get_stem(w1))
            stemming_cache.update({w1:stemmed_w1})

            w2 = words[idx+1].strip().lower()
            stemmed_w2 = stemming_cache.get(w2,get_stem(w2))
            stemming_cache.update({w2:stemmed_w2})

            tup = tuple(sorted([stemmed_w1,stemmed_w2]))
            if(w1 != "" and w2 != "" and (not w1 in words_to_remove) and (not w2 in words_to_remove)):# and tup in all_vocab):
                bigrams.append(tup)

    log_prob = {}
    for label in class_vocab: 
        prior = class_doc_counts[label]/sum(class_doc_counts.values())
        log_posteriors = []
        for gram in bigrams:
            log_posteriors.append(compute_log_posterior(gram, label))
        log_prob.update({label:np.sum(log_posteriors)+np.log(prior)})
    
    return max(log_prob, key=log_prob.get)

def compute_accuracy(y_act, y_pred):
    correct = 0
    for i in range(len(y_act)):
        if(y_act[i]==y_pred[i]):
            correct += 1
    
    return round(100*correct/len(y_act), 4)


def get_confusion_metrics(y_act, y_pred):
    conf_mat = [ [0] * 5 for _ in range(5)]
    for i,j in zip(y_act, y_pred):
        conf_mat[int(i)-1][int(j)-1] += 1
    return conf_mat

def compute_f1(conf_mat):
    class_f1 = []
    for i in range(len(conf_mat[0])):
        
        den1 = np.sum([conf_mat[j][i] for j in range(len(conf_mat[0]))])
        den2 = np.sum([conf_mat[i][j] for j in range(len(conf_mat[0]))])
        if(den1 == 0 or den2 == 0):
            class_f1.append(0)
            continue
        
        prec = (conf_mat[i][i])/den1
        rec = (conf_mat[i][i])/den2
        if(prec == 0 or rec == 0):
            class_f1.append(0)
            continue
        
        f1 = round((2*prec*rec*100)/(prec+rec),1)
        class_f1.append(f1)
            
    return class_f1


if __name__=="__main__":

    start_time = time.time()

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    data_json = []
    with open(train_data_path) as f:
        for line in f:
            data_json.append(json.loads(line))

    fin = open(f"bigram_stemmed_wdout_stop_vocab.json","r")
    class_vocab = json.load(fin)
    class_vocab = {b : dict([tuple(a.split(",")), int(x)] for a, x in class_vocab[b].items()) for b in class_vocab.keys()}
    all_vocab = list()
    for c in class_vocab:
        all_vocab.extend(class_vocab[c].keys())
    all_vocab = set(all_vocab)
    class_word_count = {label : sum(class_vocab[label].values()) for label in class_vocab}


    data = pd.DataFrame(data_json)
    review_text_data = pd.DataFrame(data).reviewText.values.tolist()
    class_label = data.overall.values.tolist()
    class_doc_counts = dict(data.overall.value_counts())
    class_doc_counts = {str(label) : class_doc_counts[label] for label in class_doc_counts}

    training_pred = []
    for review in review_text_data:
        training_pred.append(float(predict_class_bigram(review)))

    training_acc = compute_accuracy(class_label[:],training_pred)
    training_cf = get_confusion_metrics(training_pred, class_label)

    print(f"Training accuracy : {training_acc}")
    print(training_cf)

    test_data_json = []
    with open(test_data_path) as f:
        for line in f:
            test_data_json.append(json.loads(line))
    test_data = pd.DataFrame(test_data_json)
    test_reviews = test_data.reviewText.values.tolist()
    test_labels = test_data.overall.values.tolist()

    test_pred = []
    for review in test_reviews:
        test_pred.append(float(predict_class_bigram(review)))

    test_acc = compute_accuracy(test_pred, test_labels)
    test_cf = get_confusion_metrics(test_pred, test_labels)
    print(f"Test accuracy : {test_acc}")
    print(test_cf)
    print(compute_f1(test_cf))

    print(f"Total time {time.time()-start_time}")


# /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/train.csv /home/prayushi/Desktop/IITD/Assignments/ML/Assignment_2/Data/mnist/test.csv
