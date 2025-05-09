# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
from nltk.util import ngrams 

'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)

    # stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don't", "should", "now"]
    # stop_words = ["the", "a", "in", "you"]
    # for i in range(len(train_set)):
    #     train_set[i] = list(set(train_set[i]).difference(stop_words))

    # for i in range(len(dev_set)):
    #     dev_set[i] = list(set(dev_set[i]).difference(stop_words))

    return train_set, train_labels, dev_set, dev_labels

def create_unigram_word_counts(train_set, train_labels):

    positive_word_count = {}
    negative_word_count = {}
    
    total_positive_words = 0
    total_negative_words = 0

    for i in range(len(train_labels)):
        word_list = train_set[i]
        label = train_labels[i]
        x = Counter(word_list)
        for key, value in x.items():
            if label:
                positive_word_count[key] = positive_word_count.get(key, 0) + value
                total_positive_words += value
            else:
                negative_word_count[key] = negative_word_count.get(key, 0) + value
                total_negative_words += value
    
    return positive_word_count, negative_word_count, total_positive_words, total_negative_words

def create_bigram_word_counts(train_set, train_labels):

    positive_pair_count = {}
    negative_pair_count = {}
    
    total_positive_pairs = 0
    total_negative_pairs = 0

    for i in range(len(train_labels)):
        word_list = train_set[i]
        label = train_labels[i]
        n_gram = 2
        x = Counter(ngrams(word_list, n_gram))
        for key, value in x.items():
            if label:
                positive_pair_count[key] = positive_pair_count.get(key, 0) + value
                total_positive_pairs += value
            else:
                negative_pair_count[key] = negative_pair_count.get(key, 0) + value
                total_negative_pairs += value
    
    return positive_pair_count, negative_pair_count, total_positive_pairs, total_negative_pairs

def calculate_unigram_score(pos_prior, neg_prior, doc, positive_unigram_unk, negative_unigram_unk, positive_review_word_counts, unigram_laplace, negative_review_word_counts, positive_unigram_denominator, negative_unigram_denominator):
    positive_score = math.log(pos_prior)
    negative_score = math.log(neg_prior)

    for word in doc:

        positive_prior_word = positive_unigram_unk
        negative_prior_word = negative_unigram_unk

        if word in positive_review_word_counts:
            positive_prior_word = (positive_review_word_counts[word] + unigram_laplace) / positive_unigram_denominator  

        if word in negative_review_word_counts:
            negative_prior_word = (negative_review_word_counts[word] + unigram_laplace) / negative_unigram_denominator

        positive_score += math.log(positive_prior_word)
        negative_score += math.log(negative_prior_word)

    return (positive_score, negative_score)

def calculate_bigram_score(pos_prior, neg_prior, doc, positive_bigram_unk, negative_bigram_unk, positive_review_pair_counts, bigram_laplace, negative_review_pair_counts, positive_bigram_denominator, negative_bigram_denominator):
    positive_score = math.log(pos_prior)
    negative_score = math.log(neg_prior)

    for i in range(len(doc) - 1):
        pair = (doc[i], doc[i + 1])

        positive_prior_word = positive_bigram_unk
        negative_prior_word = negative_bigram_unk

        if pair in positive_review_pair_counts:
            positive_prior_word = (positive_review_pair_counts[pair] + bigram_laplace) / positive_bigram_denominator  

        if pair in negative_review_pair_counts:
            negative_prior_word = (negative_review_pair_counts[pair] + bigram_laplace) / negative_bigram_denominator

        positive_score += math.log(positive_prior_word)
        negative_score += math.log(negative_prior_word)
    
    return (positive_score, negative_score)

"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=.005, bigram_laplace=.004, bigram_lambda=.51, pos_prior=0.79, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # unigram values
    positive_review_word_counts, negative_review_word_counts, total_positive_words, total_negative_words = create_unigram_word_counts(train_set, train_labels)
#.8035
    positive_unigram_v = len(positive_review_word_counts.keys())
    negative_unigram_v = len(negative_review_word_counts.keys())

    positive_unigram_denominator = (total_positive_words + unigram_laplace * (positive_unigram_v + 1))
    negative_unigram_denominator = (total_negative_words + unigram_laplace * (negative_unigram_v + 1))

    positive_unigram_unk = unigram_laplace / positive_unigram_denominator
    negative_unigram_unk = unigram_laplace / negative_unigram_denominator

    # bigram values
    positive_review_pair_counts, negative_review_pair_counts, total_positive_pairs, total_negative_pairs = create_bigram_word_counts(train_set, train_labels)
    positive_bigram_v = len(positive_review_pair_counts.keys())
    negative_bigram_v = len(negative_review_pair_counts.keys())

    positive_bigram_denominator = (total_positive_pairs + bigram_laplace * (positive_bigram_v + 1))
    negative_bigram_denominator = (total_negative_pairs + bigram_laplace * (negative_bigram_v + 1))

    positive_bigram_unk = bigram_laplace / positive_bigram_denominator
    negative_bigram_unk = bigram_laplace / negative_bigram_denominator

    # both values
    neg_prior = float(1 - pos_prior)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):

        positive_unigram_score, negative_unigram_score = calculate_unigram_score(pos_prior, neg_prior, doc, positive_unigram_unk, negative_unigram_unk, positive_review_word_counts, unigram_laplace, negative_review_word_counts, positive_unigram_denominator, negative_unigram_denominator)

        positive_bigram_score, negative_bigram_score = calculate_bigram_score(pos_prior, neg_prior, doc, positive_bigram_unk, negative_bigram_unk, positive_review_pair_counts, bigram_laplace, negative_review_pair_counts, positive_bigram_denominator, negative_bigram_denominator)

        positive_score = positive_bigram_score
        negative_score = negative_bigram_score

        positive_score = (1 - bigram_lambda) * positive_unigram_score + bigram_lambda * positive_bigram_score
        negative_score = (1 - bigram_lambda) * negative_unigram_score + bigram_lambda * negative_bigram_score

        if positive_score >= negative_score:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



