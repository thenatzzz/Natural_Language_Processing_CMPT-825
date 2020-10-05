import os, sys, optparse
import tqdm
import numpy
from scipy.spatial.distance import cdist
import pymagnitude

import re
# from copy import deepcopy
from numpy import dot
from numpy.linalg import norm
import pandas as pd

class LexSub:

    def __init__(self, wvec_file, topn=10,lexicon=None):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        # self.wvecfile = wvec_file
        self.topn = topn
        self.lexicon = lexicon

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        # print("word: ",sentence[index])
        # print(self.wvecs.dim)
        # print(self.wvecs.most_similar(sentence[index], topn=self.topn))
        # print(self.lexicon[sentence[index]],'-------')
        # print("default: ",list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))
        new_wvecs = retrofit(self.wvecs,self.lexicon,sentence[index],num_iters=10)
        return new_wvecs[:self.topn]
        # return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))


'''Helper function'''
def retrofit(wvecs,lexicon,word,num_iters=10):
    # new_wvecs = deepcopy(wvecs)
    '''initialize new word vector'''
    new_wvecs = wvecs

    # wvec_dict = set(new_wvecs.keys())
    '''get top N words from GloVe that are most similar to word from text '''
    wvec_dict = set(map(lambda k: k[0], wvecs.most_similar(word, topn=150)))
    vector_target = wvecs.query(word)

    '''get list of mutual/intersected word between Lexicon and the N most similar words'''
    loop_dict = wvec_dict.intersection(set(lexicon.keys()))

    '''dict to store words as key and new vectors as value'''
    result_vector={}
    dict_similarity_result = {}
    ''' iterate based on number of time we want to update'''
    for iter in range(num_iters):
        '''loop through every node also in ontology (else just use data estimate)'''
        for word_sub in loop_dict:
            '''get list of neighbor words (from Lexicon) that match the top N most similar word'''
            word_neighbours = set(lexicon[word_sub]).intersection(wvec_dict)
            num_neighbours = len(word_neighbours)
            '''if words in list of mutual word do not have neighbor word, we just use estimate (no retrofit)'''
            if num_neighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            new_vec = num_neighbours * wvecs.query(word_sub)
            # loop over neighbours and add to new vector (currently with weight 1)
            for pp_word in word_neighbours:
                #new_vec += new_wvecs.query(pp_word)
            #result_vector[word_sub] = 0.6 * new_vec / (2 * num_neighbours)
                new_vec += calculate_cosine_sim(new_wvecs.query(pp_word), new_wvecs.query(word_sub))
            result_vector[word_sub] = (num_neighbours * calculate_cosine_sim(new_wvecs.query(word),
                                                                   new_wvecs.query(word_sub)) + new_vec) / 2 * num_neighbours

    '''get word vector of interested word in text'''
    vector_mainWord = wvecs.query(word)
    '''create new dict that stores calculated cosine similarity between new word vector with interested word vector '''
    dict_similarity_result = {}
    for word, vector in result_vector.items():
        dict_similarity_result[word] = calculate_cosine_sim(vector_mainWord, vector)
    '''sort result dict by similarity'''
    dict_similarity_result={k: v for k, v in sorted(dict_similarity_result.items(), key=lambda item: item[1],reverse=True)}

    '''return to list of the most similar word'''
    list_most_similar_word = list(dict_similarity_result.keys())

    return list_most_similar_word

def calculate_cosine_sim(vect1,vect2):
    return dot(vect1, vect2)/(norm(vect1)*norm(vect2))

def cosine_dis(array1,array2):
    array1 = numpy.reshape(array1, (-1, 100))
    array2 = numpy.reshape(array2,(-1,100))
    sumyy = (array2 ** 2).sum(1)
    sumxx = (array1 ** 2).sum(1, keepdims=1)
    sumxy = array1.dot(array2.T)
    result = (sumxy / numpy.sqrt(sumxx)) / numpy.sqrt(sumyy)
    result = numpy.squeeze(result)
    return result

''' Read the Lexicon (word relations) as a dictionary '''
isNumber = re.compile(r'\d+.*')
def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()

def read_lexicon(filename):
    lexicon = {}
    for line in open("/Users/wusiyu/Desktop/nlp-class-hw/lexsub/data/lexicons/wordnet-synonyms.txt", 'r',encoding='utf-8'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('/Users','wusiyu','Desktop','nlp-class-hw','lexsub','data', 'input', 'dev.txt'), help="input file with target word in context")
    #optparser.add_option("-i", "--inputfile", dest="input", default=os.path("/Users/wusiyu/Desktop/nlp-class-hw/lexsub/data/input/dev.txt"), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('/Users','wusiyu','Desktop','nlp-class-hw','lexsub','answer','data', 'glove.6B.100d.magnitude'), help="word vectors file")
    #optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path("/Users/wusiyu/Desktop/nlp-class-hw/lexsub/answer/data/glove.6B.100d.magnitude"), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-L", "--lexiconfile", dest="lexicon", default=os.path.join('/Users','wusiyu','Desktop','nlp-class-hw','lexsub','data', 'lexicons','wordnet-synonyms.txt'), help="lexicon file")
    #optparser.add_option("-L", "--lexiconfile", dest="lexicon", default=os.path("/Users/wusiyu/Desktop/nlp-class-hw/lexsub/data/lexicons/wordnet-synonym.txt"), help="lexicon file")


    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexicon = read_lexicon(opts.lexicon)
    # wvecs = read_glove(opts.wordvecfile)
    # print(wvecs)
    lexsub = LexSub(opts.wordvecfile, int(opts.topn),lexicon)

    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
        # for line in f:

            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
            # print('\n\n\n')
