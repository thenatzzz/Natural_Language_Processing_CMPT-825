import os, sys, optparse
import tqdm
import pymagnitude

import re
from copy import deepcopy

class LexSub:

    def __init__(self, wvec_file, topn=10,lexicon=None):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn
        self.lexicon = lexicon

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        print("word: ",sentence[index])
        # print(len(self.wvecs))
        # print(self.wvecs.dim)
        # print(self.wvecs.most_similar(sentence[index], topn=self.topn))
        # print(self.wvecs.query(sentence[index]))
        # print(self.lexicon[sentence[index]],'-------')
        print("default: ",list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))
        new_wvecs = retrofit(self.wvecs,self.lexicon,sentence[index],num_iters=10)

        return new_wvecs
        # return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

'''Helper function'''
def retrofit(wvecs,lexicon,word,num_iters=10):
    # new_wvecs = deepcopy(wvecs)
    new_wvecs = wvecs

    # wvec_dict = set(new_wvecs.keys())
    # wvec_dict = set(map(lambda k: k[0], wvecs.most_similar(word, topn=2000)))
    wvec_dict = set(map(lambda k: k[0], wvecs.most_similar(word, topn=200)))

    loop_dict = wvec_dict.intersection(set(lexicon.keys()))

    result={}
    for iter in range(num_iters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loop_dict:
            word_neighbours = set(lexicon[word]).intersection(wvec_dict)
            num_neighbours = len(word_neighbours)
            #no neighbours, pass - use data estimate
            if num_neighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            new_vec = num_neighbours * wvecs.query(word)
            # loop over neighbours and add to new vector (currently with weight 1)
            for pp_word in word_neighbours:
                new_vec += new_wvecs.query(pp_word)
            result[word]=new_vec/(2*num_neighbours)
    # print(pymagnitude.Magnitude(result))
    print(len(result))

    from numpy import dot
    from numpy.linalg import norm

    vector_mainWord = wvecs.query(word)
    dict_similarity_result= {}
    for word,vector in result.items():

        cos_sim = dot(vector_mainWord, vector)/(norm(vector_mainWord)*norm(vector))
        dict_similarity_result[word] = cos_sim
        # print(cos_sim)
    # sort dict by val
    dict_similarity_result={k: v for k, v in sorted(dict_similarity_result.items(), key=lambda item: item[1],reverse=True)}
    n_items = list(dict_similarity_result.keys())[:10]

    print(n_items)
    return n_items
''' Read the PPDB word relations as a dictionary '''
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
    for line in open(filename, 'r',encoding='utf-8'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-L", "--lexiconfile", dest="lexicon", default=os.path.join('data', 'input', 'lexicon','fragment.txt'), help="lexicon file")

    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexicon = read_lexicon(opts.lexicon)
    # print(lexicon['side'])
    lexsub = LexSub(opts.wordvecfile, int(opts.topn),lexicon)
    # lexsub = LexSub('answer/data/glove.6B.100d.magnitude', int(opts.topn))

    num_lines = sum(1 for line in open(opts.input,'r'))

    i = 0
    with open(opts.input) as f:
        # for line in tqdm.tqdm(f, total=num_lines):
        for line in f:
            # print("line: ",line)
            print("line number: ",i)
            fields = line.strip().split('\t')
            print(fields)
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
            print('\n')
            if i==12:
                break
            i += 1
