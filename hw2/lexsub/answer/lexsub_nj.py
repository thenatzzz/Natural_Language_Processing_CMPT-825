#!/usr/bin/env python
import os, sys, optparse
import tqdm
import pymagnitude

import re
from copy import deepcopy
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import numpy as np

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        word = sentence[index]

        # WINDOW = 2
        # list_context = get_context_word(sentence,index,window=WINDOW)

        # new_wvecs = retrofit(self.wvecs,self.lexicon,word,list_context,num_iters=1)

        # return new_wvecs[:self.topn]
        return (list(map(lambda k: k[0], self.wvecs.most_similar(word, topn=self.topn))))

def retrofit(wvecs, lexicon, iteration=10):
    '''copy val from old vec to new word vector'''
    new_wvecs = deepcopy(wvecs)

    '''get unique vocab names'''
    wvec_dict = set(new_wvecs.keys())

    '''get list of mutual/intersected word between Lexicon and the embedding keys/words'''
    loop_vocab = wvec_dict.intersection(set(lexicon.keys()))

    ''' iterate based on number of time we want to update'''
    for iter in range(iteration):
        '''loop through every node also in ontology (else just use estimation)'''
        for word in loop_vocab:
            '''get list of neighbor words (from Lexicon) that match the top N most similar word'''
            word_neighbours = set(lexicon[word]).intersection(wvec_dict)
            num_neighbours= len(word_neighbours)

            if num_neighbours == 0:
                continue
            '''create new vec and estimate new vector according to neighbors'''
            new_vec = num_neighbours * wvecs[word]
            '''iterate every neighbor word and calculate according to equation'''
            for pp_word in word_neighbours:
                new_vec += new_wvecs[pp_word]
            new_wvecs[word] = new_vec/(2*num_neighbours)
    return new_wvecs


def retrofit_old(wvecs,lexicon,word,list_context,num_iters=10):

    '''initialize new word vector'''
    new_wvecs = wvecs

    '''get top N words from GloVe that are most similar to word from text '''
    wvec_dict = set(map(lambda k: k[0], wvecs.most_similar(word, topn=150)))
    # wvec_dict = set(map(lambda k: k[0], wvecs.most_similar(word, topn=500)))


    '''get list of mutual/intersected word between Lexicon and the N most similar words'''
    loop_dict = wvec_dict.intersection(set(lexicon.keys()))

    '''dict to store words as key and new vectors as value'''
    result_vector={}

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

            ## loop over neighbours and add to new vector (currently with weight 1)
            for pp_word in word_neighbours: # lexical synonym
                new_vec += new_wvecs.query(pp_word)
            result_vector[word_sub]=new_vec/(2*num_neighbours)

                # new_vec += calculate_cosine_sim(new_wvecs.query(pp_word), new_wvecs.query(word_sub))
            # result_vector[word_sub] = (num_neighbours * calculate_cosine_sim(new_wvecs.query(word),
                                                               # new_wvecs.query(word_sub)) + new_vec) / 2 * num_neighbours

            # sum_context_lexcicalSyn = 0
            # num_context_word = len(list_context)
            #
            # lexical_adjustment = 0 # BalAdd
            # for pp_word in word_neighbours: # lexical synonym
            #     new_vec += new_wvecs.query(pp_word)
            #
            #     lexical_adjustment =num_context_word * calculate_cosine_sim(new_wvecs.query(pp_word),new_wvecs.query(word_sub))
            #     for context_word in list_context:
            #         sum_context_lexcicalSyn += calculate_cosine_sim(new_wvecs.query(pp_word),new_wvecs.query(context_word))
            #     lexical_adjustment = (lexical_adjustment+ sum_context_lexcicalSyn)/(2*num_context_word)
            #
            # # first eq is retrofitting, second is lexical adjustment
            # result_vector[word_sub]=(new_vec/(2*num_neighbours)) + lexical_adjustment


    '''get word vector of interested word in text'''
    vector_mainWord = wvecs.query(word)
    '''create new dict that stores calculated cosine similarity between new word vector with interested word vector '''
    dict_similarity_result= {}
    for word,vector in result_vector.items():
        dict_similarity_result[word] = calculate_cosine_sim(vector_mainWord, vector)


    '''sort result dict by similarity'''
    dict_similarity_result={k: v for k, v in sorted(dict_similarity_result.items(), key=lambda item: item[1],reverse=True)}

    '''return to list of the most similar word'''
    list_most_similar_word = list(dict_similarity_result.keys())

    return list_most_similar_word

'''Helper function'''

def calculate_cosine_sim(vect1,vect2):
    return dot(vect1, vect2)/(norm(vect1)*norm(vect2))

def get_context_word(sentence,index,window=5):
    # check index at starting sentence
    if index-window < 0:
        str_idx = 0
    else:
        str_idx = index-window

    # check index at ending sentence
    if index+window >= len(sentence):
        end_idx =len(sentence)-1
    else:
        end_idx = index+window

    list_context = []
    for idx in range(str_idx,end_idx+1):
        if idx == index:
            continue
        list_context.append(sentence[idx])
    return list_context

''' Read the Lexicon (word relations) as a dictionary '''
is_number = re.compile(r'\d+.*')
def norm_word(word):
    if is_number.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()

def read_lexicon(filename):
    lexicon = {}
    for line in open(filename, 'r',encoding='utf-8-sig'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

def save_embedding(retrofitted_vec, saved_file):
    ''' Write word vectors to file '''
    output_file= open(saved_file, 'w',encoding='utf-8')

    for word, values in retrofitted_vec.items():
        output_file.write(word+' ')

        for val in retrofitted_vec[word]:
            output_file.write('%.4f' %(val)+' ')

        output_file.write('\n')

    output_file.close()

def load_Glove_to_dict(file_glove):
    print("Start loading Glove Model")
    file = open(file_glove,'r',encoding='utf-8')
    glove_dict = {}
    for line in tqdm.tqdm(file):
        split_lines = line.split()
        word = split_lines[0]
        word_embedding = np.array([float(value) for value in split_lines[1:]])
        glove_dict[word] = word_embedding
    print(len(glove_dict)," words of Glove loaded successful!")
    return glove_dict

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    # optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.txt'), help="word vectors file")

    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-L", "--lexiconfile", dest="lexicon", default=os.path.join('data', 'lexicons','wordnet-synonyms.txt'), help="lexicon file")
    optparser.add_option("-r", "--retrofitted_vecfile", dest="retrofitted_vecfile", default=os.path.join('data', 'retrofitted_glove.magnitude'), help="load retrofited embedding")

    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    '''get lexicon and format it'''
    lexicon = read_lexicon(opts.lexicon)

    ''' if we don't have retrofitted file, we retrofit else we just load.
        We retrofitted Glove then save its embedding for later use (txt file) '''
    if os.path.isfile(opts.retrofitted_vecfile) == False:

        print("\nRetrofitted embedding file does not exist. Let's retrofit !\n")

        glove_dict = load_Glove_to_dict(opts.wordvecfile)
        retrofitted_vec = retrofit(glove_dict,lexicon, iteration=10)

        file_loc = 'data/retrofitted_glove.txt'
        save_embedding(retrofitted_vec,file_loc)

        print('\nSuccessfully retrofitting embedding! and save to {}.'.format(file_loc))
        print('-'*50)
        print("PLEASE run pymagnitude.converter to convert .txt file to .magnitude file")
        print('-'*50)

    else:

        lexsub = LexSub(opts.retrofitted_vecfile, int(opts.topn))

        num_lines = sum(1 for line in open(opts.input,'r',encoding='utf-8-sig'))

        with open(opts.input,encoding='utf-8-sig') as f:
            # for line in tqdm.tqdm(f, total=num_lines):
            for line in f:
                fields = line.strip().split('\t')
                print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
