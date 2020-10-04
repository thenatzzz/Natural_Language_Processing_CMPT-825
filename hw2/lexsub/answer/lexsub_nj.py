import os, sys, optparse
import tqdm
import pymagnitude

import re

class LexSub:

    def __init__(self, wvec_file, topn=10,lexicon=None):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn
        self.lexicon = lexicon

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        print("word: ",sentence[index])
        print(len(self.wvecs))
        # print(self.wvecs.dim)
        # print(self.wvecs.most_similar(sentence[index], topn=self.topn))
        print(self.wvecs.query(sentence[index]))
        new_wvecs = self.retrofit(sentence[index],num_iters=10)
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

    def retrofit(self,num_iters=10):
        new_wvecs = deepcopy(self.wvecs)
        wvec_dict = set(new_wvecs.keys())
        loopVocab = wvec_dict.intersection(set(self.lexicon.keys()))
        for iter in range(num_iters):
            # loop through every node also in ontology (else just use data estimate)
            for word in loopVocab:
                wordNeighbours = set(self.lexicon[word]).intersection(wvec_dict)
                numNeighbours = len(wordNeighbours)
                #no neighbours, pass - use data estimate
                if numNeighbours == 0:
                    continue
                # the weight of the data estimate if the number of neighbours
                newVec = numNeighbours * self.wvecs[word]
                # loop over neighbours and add to new vector (currently with weight 1)
                for ppWord in wordNeighbours:
                    newVec += newWordVecs[ppWord]
        new_wvecs[word] = newVec/(2*numNeighbours)
        return new_wvecs
'''Helper function'''
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
    print(lexicon['side'])
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
