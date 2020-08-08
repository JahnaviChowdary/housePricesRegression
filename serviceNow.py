import pandas as pd
fileName = '/home/jahnavi/Downloads/Interview_Identify_similar_sentences - Questions.csv'

file = open(fileName, "r")
### reading lines as a list ###
dataList = file.readlines()

import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re

tokenizer = ToktokTokenizer()
# nltk.download('stopwords')
stopwordList = nltk.corpus.stopwords.words('english')
stopwordList.remove('no')
stopwordList.remove('not')

import spacy
nlp = spacy.load('en', parse=True, tag=True, entity=True)

def get_cosine_sim(*strs):
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)

def lemmatizeText(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def get_vectors(dataList):
    # text = [t for t in strs]
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(dataList).toarray()

def removeStopwords(text, isLowerCase=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if isLowerCase:
        filteredTokens = [token for token in tokens if token not in stopwordList]
    else:
        filteredTokens = [token for token in tokens if token.lower() not in stopwordList]
    filteredText = ' '.join(filteredTokens)
    return filteredText

### removed special characters using regex, removed stopWords using nltk TotoTokenizer and lemmatized text using spacy ###
def normalizeCorpus(corpus):
    normalizedCorpus = []
    for doc in corpus:
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        doc = re.sub(' +', ' ', doc)
        doc = removeStopwords(doc)
        doc = lemmatizeText(doc)
        normalizedCorpus.append(doc)
    return normalizedCorpus

### getting duplicate indices of sentences ###
def treatDup(sentences):
    sMap = {}
    indices = collections.defaultdict(list)
    for i, sentence in enumerate(sentences):
        if not sentence in sMap.keys():
            sMap[sentence] = i
        else:
            if len(indices[sentence]) == 0:
                indices[sentence] += [sMap[sentence]]
            indices[sentence] += [i]
    return (sMap.keys(), indices, sMap)

dataDup, indices, sMap = treatDup(dataList)
print(indices)

dupIndices = pd.DataFrame(columns=['dupIndices'])
for i, sentence in enumerate(dataList):
    if len(indices[sentence]) > 0:
        print(indices[sentence])
        print(i)
        row = []
        row += indices[sentence]
        print(row)
        row.remove(i)
        print(row)
        dupIndices.loc[i] = str(row)
    else:
        dupIndices.loc[i] = ""

print(dupIndices)
dupIndices.to_csv('/home/jahnavi/Downloads/serviceNow/dupInd.csv', index=True)

dataNorm = normalizeCorpus(dataDup)
print(len(dataNorm))

### vectorized sentences using CountVectorizer ###
vec = get_vectors(dataNorm)

df = pd.DataFrame(vec)

print(df.shape)

### transformed vectors using minMaxScaler ###
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scalerFix = scaler.fit(df)
dfTransform = scalerFix.transform(df)
print(dfTransform.max(axis = 0) )
print(dfTransform.min(axis = 0) )

print(dfTransform)

import math
def get_cosine(vec1, vec2):
    # intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in range(len(vec1))])

    sum1 = sum([vec1[x] ** 2 for x in range(len(vec1))])
    sum2 = sum([vec2[x] ** 2 for x in range(len(vec1))])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def getCos2D(df):
    n = len(df)
    cos = [[0]*n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if j > i:
                cos[i][j] = get_cosine(df[i], df[j])
            elif i == j:
                cos[i][j] = 1.0
            else:
                cos[i][j] = cos[j][i]
    return cos

### cosine_similarity of vectors ###
cos = cosine_similarity(dfTransform)
cos2D = getCos2D(dfTransform)
print(cos)

### clustered sentences using KMeans ###
def clusterSentences(sentences, nClusters=5):
    vectorizer = CountVectorizer()
    vec = vectorizer.fit_transform(sentences).toarray()
    kmeans = KMeans(n_clusters=nClusters)
    kmeans.fit(vec)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return dict(clusters)

nClusters = 729
clusters = clusterSentences(dataNorm, nClusters)

### grouping all similar indices from the clusters formed ###
similarIndices = collections.defaultdict(list)
for cluster in range(nClusters):
    print("cluster ", cluster, ":")
    for i, sentence in enumerate(clusters[cluster]):
        print("\tsentence", i, ": ", list(dataDup)[sentence])
        senStr = list(dataDup)[sentence]
        similarIndices[cluster] += [sMap[senStr]]

print(similarIndices.values())

similarID = pd.DataFrame(columns=['similarID'])
for i in range(len(dataList)):
    similarID.loc[i] = ""
    for cluster in range(nClusters):
        ind = similarIndices[cluster]
        if i in ind:
            row = []
            row += ind
            print(row)
            row.remove(i)
            print(row)
            similarID.loc[i] += str(row)
            break

print(similarID)
similarID.to_csv('/home/jahnavi/Downloads/serviceNow/similarID.csv', index=False)

# dupDf = pd.read_csv('/home/jahnavi/Downloads/serviceNow/dupInd.csv')
# similarDf = pd.read_csv('/home/jahnavi/Downloads/serviceNow/similarID.csv')
#
# print(similarDf)
# dupSimilarJoin = dupDf.join(similarDf, on='ID', how='inner')

### getting top 2 similar sentences and their similarity score based on cosine_similarity from the clusters formed earlier ###
import ast
similarity = pd.DataFrame(columns=['similarity'])
similarInd = pd.DataFrame(columns=['similarInd'])
print(similarID.loc[0][0])
sim = ast.literal_eval(similarID.loc[0][0])
print(max(sim))

for i in range(len(dataList)):
    similarity.loc[i] = ""
    similarInd.loc[i] = ""
    if similarID.loc[i][0] and len(ast.literal_eval(similarID.loc[i][0])) > 0:
        row = []
        for ind in ast.literal_eval(similarID.loc[i][0]):
            iDup = list(dataDup).index(dataList[i])
            indDup = list(dataDup).index(dataList[ind])
            row += [cos[iDup][indDup]]
        max1 = max(row)
        ind1 = ast.literal_eval(similarID.loc[i][0])[row.index(max1)]
        row[row.index(max1)] = -1
        if len(row) > 0:
            max2 = max(row)
            ind2 = ast.literal_eval(similarID.loc[i][0])[row.index(max2)]
            similarity.loc[i] += str([max1, max2])
            similarInd.loc[i] += str([ind1, ind2])
        else:
            similarity.loc[i] += str([max1])
            similarInd.loc[i] += str([ind1])

print(similarity)
print(similarInd)

similarity.to_csv('/home/jahnavi/Downloads/serviceNow/similarity.csv', index=False)
similarInd.to_csv('/home/jahnavi/Downloads/serviceNow/similarInd.csv', index=False)