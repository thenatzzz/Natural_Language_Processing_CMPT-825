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
        print(self.lexicon[sentence[index]],'-------')
        print(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))
        new_wvecs = retrofit(self.wvecs,self.lexicon,sentence[index],num_iters=10)
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

'''Helper function'''
def retrofit(wvecs,lexicon,word,num_iters=10):
    # new_wvecs = deepcopy(wvecs)
    new_wvecs = wvecs

    # wvec_dict = set(new_wvecs.keys())
    # wvec_dict = new_wvecs.query(word)
    # wvec_dict = word

    # print(wvec_dict)
    # print(len(set(lexicon.keys())))
    # print(len(lexicon.keys()))
    print(wvecs.query(word))

    # loop_dict = wvec_dict.intersection(set(lexicon.keys()))
    loop_dict = []
    for lexicon_key in set(lexicon.keys()):
        # if lexicon_key in wvecs.most_similar(word, 20):
        if lexicon_key in wvecs:

            loop_dict.append(lexicon_key)
    print(loop_dict,' ++++++++++')
    for iter in range(num_iters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loop_dict:
            word_neighbours = set(lexicon[word]).intersection(wvec_dict)
            num_neighbours = len(word_neighbours)
            #no neighbours, pass - use data estimate
            if num_neighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            new_vec = num_neighbours * wvecs[word]
            # loop over neighbours and add to new vector (currently with weight 1)
            for pp_word in word_neighbours:
                new_vec += newWordVecs[pp_word]
                new_wvecs[word] = new_vec/(2*num_neighbours)
    return new_wvecs
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
            if i==2:
                break
            i += 1
