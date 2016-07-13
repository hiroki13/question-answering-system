import gzip
import math
import numpy as np


def load_data_sample(fn):
    samples = []
    with gzip.open(fn, 'rb') as gf:
        for line in gf:
            line = line.rstrip().split("\t")
            if len(line) < 3:
                pass
            else:
                samples.append(line[-1].split())
    return samples


def load(path):
    corpus = []
    with gzip.open(path) as f:
        for line in f:
            line = line.rstrip().split('\t')
            if len(line) == 3:
                corpus.append(line[1].split())
    return corpus


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/float(len(union))


def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)


def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([token for sent in tokenized_documents for token in sent])
    for token in all_tokens_set:
        contains_token = map(lambda sent: token in sent, tokenized_documents)
        idf_values[token] = 1 + math.log(len(tokenized_documents)/float(sum(contains_token)))
    return idf_values


def tfidf(documents):
    idf = inverse_document_frequencies(documents)
    tfidf_documents = []
    for sent in documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, sent)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return np.asarray(tfidf_documents, dtype='float32')

"""
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude = np.sqrt(np.sum(vector1 ** 2)) * np.sqrt(np.sum(vector2 ** 2))
#    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude
"""

def cosine_similarity(matrix1, matrix2):
    dot_product = np.dot(matrix1, matrix2.T)  # n_docs1 * n_docs2
    magnitude1 = np.sqrt(np.sum(np.square(matrix1), 1))
    magnitude2 = np.sqrt(np.sum(np.square(matrix2), 1))
    sims = []
    for i, m1 in enumerate(magnitude1):
        for j, m2 in enumerate(magnitude2):
            sims.append((dot_product[i][j] / (m1 * m2), i, j))
    return sims


def main(argv):
    tr_doc = load(argv.train_data)  # QA data
    test_doc = load_data_sample(argv.test_data)  # Tokenized Ubuntu dialog data
    print 'TRAIN DOC: %d\tTEST DOC: %d\n' % (len(tr_doc), len(test_doc))

#    sims = []
#    for i, doc1 in enumerate(tr_doc):
#        for j, doc2 in enumerate(test_doc):
#            sim = jaccard_similarity(doc1, doc2)
#            sims.append((sim, i, j))

    print 'Building TF-IDF Matrix...'
    tfidf_docs = tfidf(tr_doc + test_doc)
    tfidf1 = tfidf_docs[:len(tr_doc)]
    tfidf2 = tfidf_docs[len(tr_doc):]
    print 'Computing Cosine Similarity...'
    print tfidf_docs.shape
    print tfidf1.shape
    sims = cosine_similarity(tfidf1, tfidf2)

    for x in sorted(sims, reverse=True):
        print x
        print '%s\n%s\n' % (tr_doc[x[1]], test_doc[x[2]])
