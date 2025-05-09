"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""


import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect
alpha = 1e-5 

def helper(word):
    '''
    input:  word
    output: classification
    '''

    # starts and ends with a numerical digit
    if word[0].isdigit() and word[-1].isdigit():
        return "NUMERICAL_DIGIT"
    elif len(word) >= 1 and len(word) <= 3:
        return "VERY_SHORT"
    elif len(word) >= 4 and len(word) <= 9:
        if word[-1] == 's':
            return "SHORT_S"
        else:
            return "SHORT_OTHER"
    elif len(word) >= 10:
        if word[-1] == 's':
            return "LONG_S"
        else:
            return "LONG_OTHER"
    
def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}

    count_total = defaultdict(lambda: 0) # {tag: (total)}

    tag_count = defaultdict(lambda: 0)

    tag_set = set()
    
    hapax = defaultdict(lambda: defaultdict(lambda: 0))

    types = ["NUMERICAL_DIGIT", "VERY_SHORT", "SHORT_S", "SHORT_OTHER", "LONG_S", "LONG_OTHER"]

    word_counter = Counter()
    word_tag_map = defaultdict(set)
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    for sentence in sentences:
        for (word, tag) in sentence:
            word_counter[word] += 1
            word_tag_map[word].add(tag)

            tag_set.add(tag)
            count_total[tag] += 1

            init_prob[tag] = init_prob.get(tag, 0) + 1
            emit_prob[tag][word] = emit_prob[tag].get(word, 0) + 1

        for i in range(1, len(sentence)):
            (_, prev_tag) = sentence[i - 1]
            (_, curr_tag) = sentence[i]

            trans_prob[prev_tag][curr_tag] = trans_prob[prev_tag][curr_tag] + 1

            tag_count[prev_tag] += 1


    total_hapax = 0

    for word, count in word_counter.items():
        if count == 1:
            total_hapax += 1
            tag = list(word_tag_map[word])[0]
            word_type = helper(word)
            hapax[tag][word_type] = hapax[tag].get(word_type, 0) + 1


    for key in tag_set:
        for word_type in types:
            hapax[key][word_type] = hapax[key].get(word_type, 1.0) / total_hapax

            # print(trans_prob.keys())

#     print(hapax)

    # smooth emit_prob
    for tag in emit_prob.keys():

        Vt = len(emit_prob[tag])
        nt = count_total[tag]

        alpha_sum = sum(hapax[tag].get(word_type, 1.0 / total_hapax) for word_type in types)

        laplace_denom = nt + alpha_sum * alpha * (Vt + 1)
        for word in emit_prob[tag]:
            emit_prob[tag][word] = (emit_prob[tag][word] + (alpha_sum * alpha)) / laplace_denom

        # Handle unseen words by word type
        for word_type in types:
            current_alpha = hapax[tag].get(word_type, 1.0 / total_hapax) * alpha
            emit_prob[tag][word_type] = current_alpha / laplace_denom
                
    # smooth trans_prob
    for tag0 in trans_prob.keys():
        
        Vt = len(trans_prob[tag0])
        nt = tag_count[tag0]

        laplace_denom = (nt + (alpha * (Vt + 1)))

        for tag in tag_set:
            trans_prob[tag0][tag] = float((trans_prob[tag0][tag] + alpha) / laplace_denom)

        trans_prob[tag0]["UNKNOWN"] = float(alpha / laplace_denom)

    # smooth init_prop
    for key in init_prob.keys():
        if key == "START":
            init_prob[key] = 1
        else:
            init_prob[key] = epsilon_for_pt

    for key in emit_prob.keys():
        s = 0
        for word, value in emit_prob[key].items():
            s += value
        print(s)


    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """

    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    word_type = helper(word)

    # TODO: (II)

    if i == 0:
        for tag in emit_prob.keys():

            pE_unknown = emit_prob[tag].get(word_type, emit_epsilon)
            pT_unknown = trans_prob["START"].get("UNKNOWN", epsilon_for_pt)

            v = emit_prob[tag].get(word, pE_unknown)

            log_prob[tag] = math.log(v)
            predict_tag_seq[tag] = [tag]

    else:
        for tag_b in emit_prob.keys():

            tag_a_best = None
            max_prob = float('-inf')

            for tag_a in emit_prob.keys():
                
                pT_unknown = trans_prob[tag_a].get("UNKNOWN", epsilon_for_pt)
                pE_unknown = emit_prob[tag_b].get(word_type, emit_epsilon)

                pS = prev_prob[tag_a]
                pT = math.log(trans_prob[tag_a].get(tag_b, pT_unknown))
                pE = math.log(emit_prob[tag_b].get(word, pE_unknown))

                val = pS + pT + pE

                if val > max_prob:
                    max_prob = val
                    tag_a_best = tag_a

            # sequence that has gotten me to this point
            prev_seq = prev_predict_tag_seq[tag_a_best].copy()

            # i am adding on to sequence
            prev_seq.append(tag_b)

            # i am storing sequence
            predict_tag_seq[tag_b] = prev_seq

            log_prob[tag_b] = max_prob

    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    return log_prob, predict_tag_seq

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        
        # get the best tag
        best_tag = max(log_prob, key=log_prob.get)

        res = []
        ans = predict_tag_seq[best_tag]

        for i in range(len(sentence)):
            res.append((sentence[i], ans[i]))

        predicts.append(res)
        
    return predicts