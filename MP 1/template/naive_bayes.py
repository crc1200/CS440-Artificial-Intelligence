# naive_bayes.py
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
import os
import nltk, numpy
from nltk.stem import *
from nltk.stem.porter import *


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")


    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_counts(train_set, train_labels):

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


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=11, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)

    positive_review_word_counts, negative_review_word_counts, total_positive_words, total_negative_words = create_word_counts(train_set, train_labels)

    positive_v = len(positive_review_word_counts.keys())
    negative_v = len(negative_review_word_counts.keys())

    positive_denominator = (total_positive_words + laplace * (positive_v + 1))
    negative_denominator = (total_negative_words + laplace * (negative_v + 1))

    positive_unk = laplace / positive_denominator
    negative_unk = laplace / negative_denominator

    neg_prior = float(1 - pos_prior)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        
        positive_score = math.log(pos_prior)
        negative_score = math.log(neg_prior)

        for word in doc:

            positive_prior_word = positive_unk
            negative_prior_word = negative_unk

            if word in positive_review_word_counts:
                positive_prior_word = (positive_review_word_counts[word] + laplace) / positive_denominator  

            if word in negative_review_word_counts:
                negative_prior_word = (negative_review_word_counts[word] + laplace) / negative_denominator

            positive_score += math.log(positive_prior_word)
            negative_score += math.log(negative_prior_word)

        if positive_score > negative_score:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
