import re
import numpy as np
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud
from string import punctuation
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Sentiment analysis
import pprint # to print dictionaries
from nrclex import NRCLex
from scipy import stats
from numpy.linalg import norm
import numpy as np
import time


def read_hp(title_path):
    with open(title_path, "r", encoding="utf8") as book:
        lines = [line.strip() for line in book.readlines() if not (line.startswith("Page |") or line.strip() == '')]
    # Join the lines into a single string
    text = '\n'.join(lines)
    text = text.replace("\n", " ").replace("\r", "").replace("CHPT", "")
    
    return text

class CustomTokenizer:
    def __init__(self):
        self.patterns = [
            #(r'\w+', 'WORD'),         # Matches words
            #(r'\d+', 'NUMBER'),        # Matches numbers
            #(r'[.,;!?]', 'PUNCTUATION'),  # Matches common punctuation
            #(r"\b(?:\w+'\w*|\w+?n't)\b", 'abbreviation')
            (r"\b\w+'t\b|\b\w+\b|'\w+\b", "WORD")
            # (r'\b[A-Za-z]+\.(?![a-z])', 'WORD')  # Matches sequences of capitalized words (potential sentences)
            # (r'\b(?:[A-Za-z]+\.?\'?[A-Za-z]*|\w+)\b', 'WORD'),
            #(r"\b(?:\w+'\w*|(?<!\w)'(?:t|re|s|m|ll|ve)\b|\w+)\b", 'WORD')
        ]

    def tokenize(self, text):
        tokens = []
        for pattern, token_type in self.patterns:
            regex = re.compile(pattern)
            matches = regex.finditer(text)
            for match in matches:
                tokens.append((match.group(), token_type))
        return tokens
    
def preproc(text, custom_tokenizer, stop_words): 
    text = text.lower()
    text = custom_tokenizer.tokenize(text)
    text = [i[0] for i in text if i[1] == 'WORD']
    text = [i for i in text if i not in stop_words]
    text = [word for word in text if not word.startswith("'")]
    text = [i for i in text if len(i)>1]
    return text

# this function create the counts and the freq for every book
def hp_count(idx, books_stem, most_comm_tot, df_book):
    #create the dict with the frequencies for the i-th book
    xx = {key: value for key, value in Counter(books_stem[idx]).items() if key in most_comm_tot}
    count = df_book['Stems'].map(xx)
    freq  = df_book['Stems'].map(xx)/df_book['Stems'].map(xx).sum()
    # df = pd.DataFrame
    return count, freq

def hp_scores(df, idx):
    N = df[f"Count_{idx}"].sum()
    n = df[f"Count_{idx}"]
    theta1 = df['Freq_1']
    theta7 = df['Freq_7']
    return round(1/N * sum(n*np.log(theta1/theta7)), 4)

def hp_var(df, idx):
    N = df[f"Count_{idx}"].sum()
    n = df[f"Count_{idx}"]
    theta1 = df['Freq_1']
    theta7 = df['Freq_7']
    return round(1/(N * (N - 1))*(sum(n * np.log(theta1/theta7)**2) - 1/N * (sum(n * np.log(theta1/theta7))**2)), 6)

def t_test(dict_1, dict_2):
    numerator = dict_1["score"] - dict_2["score"]
    denominator = np.sqrt(dict_1["Variance"] + dict_2["Variance"])

    gdl = dict_1["N"]+dict_2["N"]-2
    t_statistic = numerator / denominator
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=gdl))
    return t_statistic, p_value

def cosine(c,m): #c=centroidi m=intera
    centroidi={el:[] for el in range(m.shape[0])}
    for i in range(m.shape[0]):
        for j in range(c.shape[0]):
            centroidi[i].append(np.dot(m[i],c[j])/(norm(m[i])*norm(c[j])))
    return(centroidi)

def centroidi(m,g): #matrice, dizionario gruppi
    centroide=np.zeros(shape=(len(g.keys()),m.shape[1]))
    for i in range(len(g.keys())):
        centroide[i]=np.sum(m[g[i]], axis=0)/len(g[i])
    return(centroide)

def kmeans(m,c): #m=matrice stilemi, c=centroidi
#con il seguente comando si ottiene un dizionario, dove per chiave c'è l'indice di riga di ciascuna trama e per valori una lista contenente le distanze della trama da ciascuno dei 26 centroidi.
    print("esecuzione...")
    k=c.shape[0]
    distanze=cosine(c,m)
#Con il seguente comando a ciascuna lista di distanze viene aggiunto l'indice del centroide che presenta la similarità massima con quella trama 
    for z in range(len(distanze.keys())):
        distanze[z]=[distanze[z],[[i for i, j in enumerate(distanze[z]) if j == max(distanze[z])][0]]]
    groups={el:[] for el in range(k)}
    centroids=list(range(k))
    for i in centroids:
        for j in distanze.keys():
            if distanze[j][1][0]==centroids[i]: groups[i].append(j)
    coesione=k*[0]
    for i in range(len(distanze.keys())):
        coesione[distanze[i][1][0]]+=distanze[i][0][distanze[i][1][0]]
    coesione.append([sum(coesione)]) #mi da la coesione totale
    
    #nuovi centroidi:
    centroids_new=centroidi(m, groups)
    distanze_new=cosine(centroids_new,m)
    for z in range(len(distanze_new.keys())):
        distanze_new[z]=[distanze_new[z],[[i for i, j in enumerate(distanze_new[z]) if j == max(distanze_new[z])][0]]]
    groups_new={el:[] for el in range(k)}
    centroids=list(range(k))
    for i in centroids:
        for j in distanze_new.keys():
            if distanze_new[j][1][0]==centroids[i]: groups_new[i].append(j) 
    coesione_new=k*[0]
    for i in range(len(distanze_new.keys())):
        coesione_new[distanze_new[i][1][0]]+=distanze_new[i][0][distanze_new[i][1][0]]
    coesione_new.append([sum(coesione_new)])
    
    b=1
    print("interazione numero {}".format(b))

    start_time = time.time()
    b=1    
    while True:
        groups=groups_new
        k_centroids=centroids_new
        coesione=coesione_new
        centroids_new=centroidi(m,groups)
        distanze_new=cosine(centroids_new,m)
        for z in range(len(distanze_new.keys())):
            distanze_new[z]=[distanze_new[z],[[i for i, j in enumerate(distanze_new[z]) if j == max(distanze_new[z])][0]]]
    
        groups_new={el:[] for el in range(k)}
        centroids=list(range(k))
        for i in centroids:
            for j in distanze_new.keys():
                if distanze_new[j][1][0]==centroids[i]: groups_new[i].append(j) 
    
        
        coesione_new=k*[0]
        for i in range(len(distanze_new.keys())):
            coesione_new[distanze_new[i][1][0]]+=distanze_new[i][0][distanze_new[i][1][0]]
        coesione_new.append([sum(coesione_new)]) 
        if coesione_new[-1][0]/coesione[-1][0]<=1:break
        else:b=b+1
        print("interazione numero {}".format(b))
    
    print("tempo impiegato: %s seconds" %(time.time() - start_time))
    print("numero interazioni: {}".format(b))
    return(groups)
    
def spectral_cl(m, k): # m = matrix, k = num clusters
    p = m.shape[1]
    s = m.shape[0]
    W = np.empty((s, s))
    for i, j in np.ndindex(W.shape):
        W[i, j] = cosine(m[i,].reshape(1, -1), m[j,].reshape(1, -1))[0][0]
    D = np.diag(np.sum(W, axis = 0))
    L = D - W
    # D^(1/2) * L * D^(1/2)
    L_sym = np.dot(np.dot(np.diag(1/np.sqrt(np.diag(D))), L), np.diag(1/np.sqrt(np.diag(D))))
    e_val, e_vec = np.linalg.eig(L_sym)
    T = U = e_vec[:, :k]
    cl = kmeans(T,T[[0, 1]])
    return cl