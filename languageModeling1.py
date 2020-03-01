import httpimport
from nltk import ngrams
import math
import operator

with httpimport.remote_repo(['lm_helper'],
                            'https://raw.githubusercontent.com/jasoriya/CS6120-PS2-support/master/utils/'):
    from lm_helper import get_train_data, get_test_data


def extractData():
    train = get_train_data()
    test, test_files = get_test_data()
    first = int(0.8 * len(train))
    training = train[:first]
    validation = train[first:]
    return train, training, validation, test, test_files


def preprocess(s):
    sample_tuple = ('^',)
    frequencies_one = {}
    one = list(ngrams(s, 1, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
    one.insert(0, sample_tuple)
    one.insert(len(one), sample_tuple)
    for u in one:
        if u[0] in frequencies_one.keys():
            frequencies_one[u[0]] += 1
        else:
            frequencies_one[u[0]] = 1
    for entry in frequencies_one.keys():
        if frequencies_one[entry] < 3:
            s = s.replace(entry, 'UNK')
    return s


# Your code here
def generateCounts(dataset, flag):
    sentence_breakdown = []
    uni_dict = {}
    bi_dict = {}
    tri_dict = {}
    bi_prob_dict = {}
    tri_prob_dict = {}
    for entry in dataset:
        each_sentence = []
        for sentence in entry:
            each_sentence.append(' '.join(sentence))
        sentence_breakdown.append(' '.join(each_sentence))

    for s in sentence_breakdown:
        s = preprocess(s)
        trigrams = list(ngrams(s, 3, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        bi_grams = list(ngrams(s, 2, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        uni_grams = list(ngrams(s, 1, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        sample_tuple = ('^',)
        uni_grams.insert(0, sample_tuple)
        uni_grams.insert(len(uni_grams), sample_tuple)
        for t in trigrams:
            if t[0] + t[1] + t[2] in tri_dict.keys():
                tri_dict[t[0] + t[1] + t[2]] += 1
            else:
                tri_dict[t[0] + t[1] + t[2]] = 1
        for b in bi_grams:
            if b[0] + b[1] in bi_dict.keys():
                bi_dict[b[0] + b[1]] += 1
            else:
                bi_dict[b[0] + b[1]] = 1
        for u in uni_grams:
            if u[0] in uni_dict.keys():
                uni_dict[u[0]] += 1
            else:
                uni_dict[u[0]] = 1
    onegram_prob_dict = computeProb(uni_dict)
    for entry in bi_dict.keys():
        if entry[0] in uni_dict.keys():
            bi_prob_dict[entry] = bi_dict[entry] / uni_dict[entry[0]]
    for entry in tri_dict.keys():
        if entry[0] + entry[1] in bi_dict.keys():
            tri_prob_dict[entry] = tri_dict[entry] / bi_dict[entry[0] + entry[1]]
    if flag == 'IN':
        return onegram_prob_dict, bi_prob_dict, tri_prob_dict
    else:
        return uni_dict, bi_dict, tri_dict


def computeProb(frequencies_dict):
    probabilities_dict = {}
    total = 0
    for k in frequencies_dict.keys():
        total += frequencies_dict[k]
    for k in frequencies_dict.keys():
        probabilities_dict[k] = frequencies_dict[k] / total
    return probabilities_dict


def heldOut(held_out_set, uni_probabilities, bi_probabilities, tri_probabilities):
    lambda_set = [[0.8, 0.1, 0.1], [0.9, 0.05, 0.05], [0.7, 0.2, 0.1], [0.6, 0.1, 0.3], [0.5, 0.3, 0.2],
                  [0.2, 0.6, 0.2], [0.1, 0.1, 0.8], [0.2, 0.2, 0.6], [0.05, 0.25, 0.7], [0.05, 0.05, 0.9]]
    minimum = 100000
    global_min = []
    sentence_breakdown = []
    for entry in held_out_set:
        individual = []
        for sentence in entry:
            individual.append(' '.join(sentence))
        sentence_breakdown.append(' '.join(individual))
    for entry in lambda_set:
        local_peplex = []
        for every in sentence_breakdown:
            perplexity = 0
            interpolated_prob_dict = {}
            sample_tuple = ('^',)
            uni = list(ngrams(every, 1, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
            uni.insert(0, sample_tuple)
            uni.insert(len(uni), sample_tuple)
            for u in uni:
                if not u[0] in uni_probabilities.keys():
                    every = every.replace(u[0], 'UNK')
            tri = list(ngrams(every, 3, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
            for token in tri:
                tri_score = 0
                bi_score = 0
                uni_score = 0
                if token[0] + token[1] + token[2] in tri_probabilities:
                    tri_score = tri_probabilities[token[0] + token[1] + token[2]]
                if token[1] + token[2] in bi_probabilities:
                    bi_score = bi_probabilities[token[1] + token[2]]
                if token[2] in uni_probabilities:
                    uni_score = uni_probabilities[token[2]]
                interpolated_prob_dict[token[0] + token[1] + token[2]] = entry[0] * uni_score + entry[1] * bi_score + \
                                                                         entry[2] * tri_score
                perplexity += math.log(interpolated_prob_dict[token[0] + token[1] + token[2]])
            final_perp = 2 ** (-1 * perplexity / len(tri))
            local_peplex.append(final_perp)
        avg_perplex = sum(local_peplex) / len(local_peplex)
        if avg_perplex < minimum:
            print('PERPLEXITY FOR LAMBDAS', entry)
            print(avg_perplex)
            minimum = avg_perplex
            global_min = entry

    print(minimum)
    print(global_min)
    return global_min


def testOnData(dataset, test_set, file_names, coeff):
    one_dict, two_dict, three_dict = generateCounts(dataset, 'IN')
    sample_tuple = ('^',)
    full_book = []
    full_book_dict = {}
    for l1 in test_set:
        book_sentences = []
        for s1 in l1:
            book_sentences.append(' '.join(s1))
        full_book.append(' '.join(book_sentences))
    for i in range(0, len(file_names)):
        perplexity = 0
        interp_dict = {}
        one_grams = list(
            ngrams(full_book[i], 1, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        one_grams.insert(0, sample_tuple)
        one_grams.insert(len(one_grams), sample_tuple)
        for entry in one_grams:
            if not entry[0] in one_dict.keys():
                full_book[i] = full_book[i].replace(entry[0], 'UNK')
        three_grams = list(
            ngrams(full_book[i], 3, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        for token in three_grams:
            one_score = 0
            two_score = 0
            three_score = 0
            if token[0] + token[1] + token[2] in three_dict:
                three_score = three_dict[token[0] + token[1] + token[2]]
            if token[1] + token[2] in two_dict:
                two_score = two_dict[token[1] + token[2]]
            if token[2] in one_dict:
                one_score = one_dict[token[2]]
            interp_dict[token[0] + token[1] + token[2]] = coeff[0] * one_score + coeff[1] * two_score + coeff[
                2] * three_score
            perplexity += math.log(interp_dict[token[0] + token[1] + token[2]])
        full_book_dict[file_names[i]] = 2 ** (-1 * perplexity / len(three_grams))
    #     full_book_list.sort(reverse=True)
    sorted_d = dict(sorted(full_book_dict.items(), key=operator.itemgetter(1), reverse=True))
    print(len(sorted_d))
    count = 0
    for k in sorted_d.keys():
        print(count, 'NAME:', k, '---', 'VALUE:', sorted_d[k])
        count += 1


def addLambdaSmoothing(dataset, test_set, file_names, coeff):
    one_dict, two_dict, three_dict = generateCounts(dataset, 'SM')
    # Your code here
    sample_tuple = ('^',)
    full_book_list = []
    full_book_dict = {}
    for l1 in test_set:
        book_sentences = []
        for s1 in l1:
            book_sentences.append(' '.join(s1))
        full_book_list.append(' '.join(book_sentences))
    for i in range(0, len(test_files)):
        perplexity = 0
        interp_dict = {}
        one_grams = list(
            ngrams(full_book_list[i], 1, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        one_grams.insert(0, sample_tuple)
        one_grams.insert(len(one_grams), sample_tuple)
        for entry in one_grams:
            if not entry[0] in one_dict.keys():
                full_book_list[i] = full_book_list[i].replace(entry[0], 'UNK')
        three_grams = list(
            ngrams(full_book_list[i], 3, pad_left=True, pad_right=True, right_pad_symbol='^', left_pad_symbol='^'))
        for token in three_grams:
            three_score = 0
            two_score = 0
            if token[0] + token[1] + token[2] in three_dict:
                three_score = three_dict[token[0] + token[1] + token[2]]
            if token[0] + token[1] in two_dict:
                two_score = two_dict[token[0] + token[1]]
            ultimate_score = (three_score + coeff) / (two_score + coeff * len(three_dict))
            interp_dict[token[0] + token[1] + token[2]] = ultimate_score
            perplexity += math.log(interp_dict[token[0] + token[1] + token[2]])
        full_book_dict[file_names[i]] = 2 ** (-1 * perplexity / len(three_grams))
    sorted_d = dict(sorted(full_book_dict.items(), key=operator.itemgetter(1), reverse=True))
    print(len(sorted_d))
    count = 0
    for k in sorted_d.keys():
        print(count, 'NAME:', k, '---', 'VALUE:', sorted_d[k])
        count += 1


train, training, validation, test, test_files = extractData()
unigram_prob_dict, bigram_prob_dict, trigram_prob_dict = generateCounts(training, 'IN')
lambdas = heldOut(validation, unigram_prob_dict, bigram_prob_dict, trigram_prob_dict)
testOnData(train, test, test_files, lambdas)
addLambdaSmoothing(train, test, test_files, 0.1)
