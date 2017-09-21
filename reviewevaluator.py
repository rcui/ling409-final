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
    from nltk.collocations import *
    bigram_measures = nltk.collocations.BigramAssocMeasures()

# Tags for adj/adv
tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']

"""
Depth to which to search for bigrams in reviews. Bigger number = more accuracy, but longer runtime.
"""
bigram_depth = 800


def get_reviews_from_corpus(category):
    """
    Gets reviews in list form from NLTK's movie_reviews corpus.

    :param category: type of reviews to return (pos or neg)
    :return: list of reviews of specified category.
    """
    return [movie_reviews.words(review) for review in movie_reviews.fileids(categories=category)]


def tag_reviews(reviews):
    """
    Tags words in reviews using NLTK's pos_tagger.

    :param reviews: reviews to tag
    :return: list of words in tuple (word, tag) form.
    """
    return [nltk.pos_tag(review) for review in reviews]


def bigramify(reviews, depth):
    """
    Function to find bigrams from a set of texts using NLTK's Collocations package.

    :param reviews: list of reviews to extract bigrams from.
    :param depth: how many bigrams to find. See bigram_depth for more information.
    :return: list of bigrams found by the function.
    """
    bigrams = []
    for review in reviews:
        finder = BigramCollocationFinder.from_words(review)
        bigrams.append(finder.nbest(bigram_measures.pmi, depth))
    return bigrams


def bigram_feats(bigrams):
    """
    Function to attach features to bigrams for the classifier. Adds all bigrams that have a adjective or adverb POS tag.

    :param bigrams: list of bigrams to process.
    :return: dictionary of bigrams with adjective/adverb features.
    """
    return dict([(bigram, True) for bigram in bigrams if bigram[0][1] in tags or bigram[1][1] in tags])


def train(positive_features, negative_features):
    """
    Function to train on movie_reviews corpus. Takes in positive and negative features and then applies ten-fold
    validation to average results.

    :param positive_features: dictionary of positive features
    :param negative_features: dictionary of negative features
    :return: list of accuracies from each validation.
    """
    train_feats, test_feats, accuracy = [], [], []
    for i in range(0, 1000, 100):
        test_feats = positive_features[i:i + 100] + negative_features[i:i + 100]
        train_feats = [pf for pf in positive_features if pf not in positive_features[i:i + 100]] + \
                      [nf for nf in negative_features if nf not in negative_features[i:i + 100]]
        classifier = NaiveBayesClassifier.train(train_feats)
        acc = nltk.classify.util.accuracy(classifier, test_feats)
        print('test set range: ', i, '-', i + 100, ' accuracy: ', acc, sep='')
        classifier.show_most_informative_features()
        print()
        accuracy.append(acc)
    return accuracy

"""
Begin program
"""

"""
Get positive/negative reviews from movie_reviews corpus
"""
pos = get_reviews_from_corpus('pos')
neg = get_reviews_from_corpus('neg')

"""
Tag reviews with POS
"""
pos_tags = tag_reviews(pos)
neg_tags = tag_reviews(neg)

"""
Get bigrams from reviews
"""
pos_b = bigramify(pos_tags, bigram_depth)
neg_b = bigramify(neg_tags, bigram_depth)


"""
Applies features to bigrams
"""
pos_feats = [(bigram_feats(bigram), 'pos') for bigram in pos_b]
neg_feats = [(bigram_feats(bigram), 'neg') for bigram in neg_b]

"""
Begin output formatting
"""

print('bigram depth: ', bigram_depth)
print()

total_accuracy = train(pos_feats, neg_feats)

print()
print('total accuracy: ', sum(total_accuracy)/len(total_accuracy))
