from __future__ import division
from __future__ import print_function

import contextlib
import sys

"""
Code to suppress console output of nltk imports.
Console output is still there, just written to a dummy file,
	so output may take a while to show up.
"""


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

"""
Suppress nltk import for better write to file.
"""
with nostdout():
    import nltk
    from nltk.corpus import movie_reviews
    from nltk.classify import NaiveBayesClassifier

# Tags for adj/adv
tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']



# All pos/neg reviews
pos = [movie_reviews.words(review) for review in movie_reviews.fileids(categories='pos')]
neg = [movie_reviews.words(review) for review in movie_reviews.fileids(categories='neg')]

def feats(words):
    return dict([(word[0], True) for word in words if word[0] in tags])

posfeats = [(feats(nltk.pos_tag(review)), 'pos') for review in pos]
negfeats = [(feats(nltk.pos_tag(review)), 'neg') for review in neg]

trainfeats = negfeats[:900] + posfeats[:900]
testfeats = negfeats[900:] + posfeats[900:]

classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats),sep=' ')
classifier.show_most_informative_features()


# # Extract adj/adv from positive reviews
# postags = [nltk.pos_tag(review) for review in pos]
# all_postags = [tag for review in postags for tag in review]
# poswords = [tag[0] for tag in all_postags if tag[1] in tags and tag[0] not in name_corpus]
#
# # Extract adj/adv from negative reviews
# negtags = [nltk.pos_tag(review) for review in neg]
# all_negtags = [tag for review in negtags for tag in review]
# negwords = [tag[0] for tag in all_negtags if tag[1] in tags and tag[0] not in name_corpus]

# # Find disjoint words from pos/neg sets
# pos_set = set(poswords) - (set(poswords) & set(negwords))
# neg_set = set(negwords) - (set(poswords) & set(negwords))
#
# # Find all pos/neg exclusive words
# pos_ex = [word for word in poswords if word in pos_set]
# neg_ex = [word for word in negwords if word in neg_set]
#
# # Find most common words in each exclusive list
# pos_common = Counter(pos_ex).most_common(100)
# neg_common = Counter(neg_ex).most_common(100)
#
# # Score text based on number of pos/neg tokens
# def score(text):
#     pos_tokens = [word for word in text if word in pos_set]
#     neg_tokens = [word for word in text if word in neg_set]
#     return len(pos_tokens)/len(text), len(neg_tokens)/len(text)
#
# def score_norm1(text):
#     pos_tokens = [word for word in text if word in pos_set]
#     neg_tokens = [word for word in text if word in neg_set]
#     return len(pos_tokens) / len(text) * len(pos_set), len(neg_tokens) / len(text) * len(neg_set)
#
# def score_norm2(text):
#     pos_tokens = [word for word in text if word in pos_set]
#     neg_tokens = [word for word in text if word in neg_set]
#     return len(pos_tokens) / len(text) * len(pos_ex), len(neg_tokens) / len(text) * len(neg_ex)
#
#
# def score_pos(case):
#     count = 0
#     for review in pos:
#         s = None
#         if case == 0:
#             s = score(review)
#         elif case == 1:
#             s = score_norm1(review)
#         elif case == 2:
#             s = score_norm2(review)
#         if s[0] < s[1]:
#             count += 1
#     return count
#
# def score_neg(case):
#     count = 0
#     for review in neg:
#         s = None
#         if case == 0:
#             s = score(review)
#         elif case == 1:
#             s = score_norm1(review)
#         elif case == 2:
#             s = score_norm2(review)
#         if s[0] > s[1]:
#             count += 1
#     return count
#
# print("No normalization:")
# print("Number of incorrect pos scores: ", score_pos(0))
# print("Number of incorrect neg scores: ", score_neg(0))
# print()
# print("Normalization by set:")
# print("Number of incorrect pos scores: ", score_pos(1))
# print("Number of incorrect neg scores: ", score_neg(1))
# print("Length of pos_set: ", len(pos_set))
# print("Length of neg_set: ", len(neg_set))
# print()
# print("No normalization by ex:")
# print("Number of incorrect pos scores: ", score_pos(2))
# print("Number of incorrect neg scores: ", score_neg(2))
# print("Length of pos_ex: ", len(pos_ex))
# print("Length of neg_ex: ", len(neg_ex))


# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# word_features = list(all_words)[:2000]
#
# def document_features(document):
#     document_words = set(document)
#     features = {}
#     for word in word_features:
#         features['contains({})'.format(word)] = (word in document_words)
#     return features
#
# featuresets = [(document_features(d), c) for (d,c) in documents]
# train_set, test_set = featuresets[100:], featuresets[:100]
# classifier = nltk.NaiveBayesClassifier.train(train_set)

