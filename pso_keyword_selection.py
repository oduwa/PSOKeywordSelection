from __future__ import print_function
from __future__ import division
import collections
import math
import numpy as np
import os
import csv,re
import random
import nltk
from nltk.corpus import stopwords
import pickle
from pyswarm import pso
import argparse
from keywords import SEED_KEYWORDS, SIMILAR_KEYWORDS
from settings import D, SWARM_SIZE, N_STEPS, OMEGA, C_1, C_2

ap = argparse.ArgumentParser(description="PSO for Automatic Keyword Selection")
ap.add_argument('-t', '--tweets', help = 'Path to a CSV file containing tweets')
args = vars(ap.parse_args())

index2keyword = {}
keyword2index = {}
tweet_list = []

CANDIDATE_KEYWORDS = SEED_KEYWORDS + SIMILAR_KEYWORDS

def load_tweets_from_csv(filepath):
    '''
    Loads tweet dictionaries from a csv table into a list. The CSV file must
    have at least two columns named "text" and "class" respectively.

    @param filepath: path to file from this directory.

    @returns: A list of tweet dictionaries. A tweet dictionary has the keys
    'text' and 'class' which are a string and an int respectively.
    '''
    global tweet_list

    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Skip past column headers
        reader.next()
        i = 0
        for row in reader:
            tweet = {}
            tweet['text'] = row[0].decode('utf-8').lower()

            # Remove user mentions
            tweet['text'] = re.sub('@[A-Za-z0-9_-]*', '<USER_MENTION>', tweet['text'])

            # remove urls
            tweet['text'] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL_HERE>', tweet['text'])

            # Remove hashtags
            #tweet = Helper.removeHashtags(tweet)

            tweet['class'] = int(row[1])

            # TODO: HANDLE EMOJIS

            tweet_list.append(tweet)
            # if i > 90000:
            #     break
            # i+=1

    return tweet_list

def irrelevance_score(keywords):
    '''
    Evaluates a given set of keywords by computing the mean of two values:
    1: irrelevance :- the proportion of tweets which contain these keywords that
        are irrelevant.
    2: collectibility inverse :-  1 - (the proportion of tweets containing the keyword).

    @param keywords A list of string keywords.
    @return Float mean(mean(irrelevance), mean(collectibility inverse)).
    '''
    global tweet_list

    # Score is infinity for no keywords
    if(len(keywords) == 0):
        return float("inf")

    kw_counts1 = [0]*len(keywords)
    kw_counts2 = [0]*len(keywords)
    irrel_count = 0
    for tweet in tweet_list:
        # Irrelevance
        if tweet['class'] == 0:
            irrel_count += 1

            for i,kw in enumerate(keywords):
                if kw in tweet['text']:
                    kw_counts1[i] += 1

        # Collectibility
        for i,kw in enumerate(keywords):
            if kw in tweet['text']:
                kw_counts2[i] += 1

    scores1 = [x/irrel_count for x in kw_counts1]
    scores2 = [1-(x/len(tweet_list)) for x in kw_counts2]
    scores = [(scores1[i]+scores2[i])/2 for i in range(len(keywords))]

    return sum(scores)/len(scores)


def construct_keyword_dictionaries():
    global keyword2index, index2keyword

    # Empty string aka no keyword
    index2keyword[0] = ""
    keyword2index[""] = 0

    # Get indices for remaining keywords
    for kw in CANDIDATE_KEYWORDS:
        idx = len(keyword2index)
        keyword2index[kw] = idx
        index2keyword[idx] = kw

    return keyword2index, index2keyword


def keyword_set_from_vector(vec):
    global keyword2index, index2keyword

    assert((len(keyword2index) == len(index2keyword)) and len(keyword2index) > 0)

    kwords = []
    for i in range(len(vec)):
        kwords.append(index2keyword[ int(round(vec[i])) ])

    # Remove empty strings before returning
    return [kw for kw in kwords if len(kw) > 0]


def objective_function(x):
    '''
    Objective function that is minimised by PSO.

    @param x list A list representing a particle, which in turn represents a set
     of keywords. Its ith position contains the ith keyword in the set.

    @Returns float value of objective function (which in this case is
    log loss)
    '''
    # Get keyword set from particle vector
    keywords = keyword_set_from_vector(x)

    # Function who's value to minimize
    return irrelevance_score(keywords)

def uniqueness_constraint(x):
    '''
    Constraint for PSO which enforces that each value in the vector be unique.

    Should return 0 or higher for a valid particle x
    '''
    y = [round(x_i) for x_i in x]
    counts = collections.Counter(y).values()
    p = 0
    for c in counts:
        p += (1-c)
    return p


def main():
    # Load data
    load_tweets_from_csv(args['tweets'])
    construct_keyword_dictionaries()

    # Dictate upper and lower bounds of a particle
    lower_bound = [0]*D
    upper_bound = [len(index2keyword)-1]*D

    print(lower_bound)
    print(upper_bound)

    # Run PSO
    xopt, fopt = pso(objective_function, lower_bound, upper_bound,
                     ieqcons=[uniqueness_constraint], swarmsize=SWARM_SIZE,
                     maxiter=N_STEPS, omega=OMEGA, phip=C_1, phig=C_2, debug=True)
    print("OPTIMAL HYPERPARAMETER SETUP: {}\nMINIMUM LOSS OBTAINED: {}".format(xopt,fopt))
    print("SELECTED KEYWORDS:\n{}".format(keyword_set_from_vector(xopt)))

if __name__ == "__main__":
    main()
