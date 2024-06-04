import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.stem import PorterStemmer

def read_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    return content

def read_stopwords(filepath):
    if not filepath:
        return set()
    with open(filepath, 'r') as file:
        stopwords = set(file.read().split())
    return stopwords

def preprocess_text(text, stopwords, stemming=False):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords]
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

class CustomTfidfVectorizer(TfidfVectorizer):
    def __init__(self, tf_method='n', idf_method='t', normalization='c', **kwargs):
        super().__init__(**kwargs)
        self.tf_method = tf_method
        self.idf_method = idf_method
        self.normalization = normalization
    
    def _tf(self, X):
        if self.tf_method == 'n':
            return X
        elif self.tf_method == 'l':
            return X.log1p()
        elif self.tf_method == 'a':
            return X
        elif self.tf_method == 'b':
            return (X > 0).astype(float)
        return X
    
    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents)
        X = self._tf(X)
        if self.idf_method == 't':
            self._idf_diag = np.diag(self.idf_)
            X = X * self._idf_diag
        if self.normalization == 'c':
            X = normalize(X, norm='l2')
        return X
    
    def transform(self, raw_documents, copy=True):
        X = super().transform(raw_documents)
        X = self._tf(X)
        if self.idf_method == 't':
            X = X * self._idf_diag
        if self.normalization == 'c':
            X = normalize(X, norm='l2')
        return X

def compute_tfidf_and_similarity(documents, queries, doc_scheme, query_scheme):
    doc_vectorizer = CustomTfidfVectorizer(tf_method=doc_scheme[0], idf_method=doc_scheme[1], normalization=doc_scheme[2])
    doc_vectors = doc_vectorizer.fit_transform(documents)
    
    query_vectorizer = CustomTfidfVectorizer(tf_method=query_scheme[0], idf_method=query_scheme[1], normalization=query_scheme[2], vocabulary=doc_vectorizer.vocabulary_)
    query_vectors = query_vectorizer.fit_transform(queries)
    
    similarities = cosine_similarity(query_vectors, doc_vectors)
    return similarities

def calculate_map(similarities, qrels):
    average_precisions = []
    for query_idx, similarity_scores in enumerate(similarities):
        relevant_docs = qrels.get(query_idx + 1, set())
        if not relevant_docs:
            continue

        ranked_docs = np.argsort(similarity_scores)[::-1]
        hits = 0
        sum_precisions = 0
        for rank, doc_idx in enumerate(ranked_docs):
            if doc_idx + 1 in relevant_docs:
                hits += 1
                precision_at_k = hits / (rank + 1)
                sum_precisions += precision_at_k

        average_precision = sum_precisions / len(relevant_docs) if relevant_docs else 0
        average_precisions.append(average_precision)

    return np.mean(average_precisions)