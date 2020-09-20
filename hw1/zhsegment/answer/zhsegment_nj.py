import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

# additional library
import operator
INDEX_PROBABILITY = 2
INDEX_WORD = 0
INDEX_BACKPOINTER = 3
class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        segmentation = [ w for w in text ] # segment each char into a word
        print('TEXT: ',text[0],self.Pw(text[0]))
        print(segmentation, len(self.Pw),Pw.N,self.Pwords(text[0]))

        segmentation = iterative_segment(text,self.Pw,self.Pwords)

        return segmentation

    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

# class Entry:
#     def __init__(self,word,start_position,log_probability,back_pointer):
#         self.word = word
#         self.start_position = start_position
#         self.log_probability =log_probability
#         self.back_pointer= back_pointer

#### Support functions (p. 224)
def iterative_segment(text,Pw,Pwords):
    print('=============== ITERATIVE SEGMENTOR =================')

    def heappush_list(h, item, key=lambda x: x):
        heapq.heappush(h, (key(item), item))
    def heappop_list(h):
        return heapq.heappop(h)[1]

    '''Initialize the HEAP'''
    heap = []
    # i= 0
    for key,value in dict(Pw).items():
        if text[0] == key[0]:
            '''multiply by -1 to get Max Heap'''
            each_entry = [key,0,-1.0*log10(Pwords(key)),None]
            heappush_list(heap, each_entry, key=operator.itemgetter(INDEX_PROBABILITY)) # sort by prob
            # heappush_list(heap, Entry(key,0,-1.0*log10(Pwords(key)),'blank'), key=operator.itemgetter(log_probability)) # sort by prob
        # i += 1

    '''Iteratively fill in CHART for all i '''
    chart = {}
    while heap:
        print('WHILE')
        '''multiply by -1 to get original value back'''
        # max probability
        entry = heappop_list(heap)
        entry[INDEX_PROBABILITY] = -1.0*entry[INDEX_PROBABILITY]
        print(entry)

        # check if list is empty, then put the first entry in
        if not chart:
            chart[0] = entry
        endindex = len(entry[INDEX_WORD])-1
        print(endindex)
        previous_entry = chart[endindex]
        if previous_entry[INDEX_BACKPOINTER] != None:
            if entry[INDEX_PROBABILITY] > previous_entry[INDEX_PROBABILITY]:
                chart[endindex] = entry
            if entry[INDEX_PROBABILITY] <= previous_entry[INDEX_PROBABILITY]:
                continue
        else:
            chart[endindex] = entry

        print(chart)

        

        #
        # break

    print('=============== END SEGMENTOR =================')

    return [ w+'_Yes' for w in text ]

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key):
        if key in self: return self[key]/self.N
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name,encoding="utf8") as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w))
    print("Pw.N: ",Pw.N, '\n\n')
    segmenter = Segment(Pw)
    i = 1
    with open(opts.input,encoding='utf8') as f:
        for line in f:
            print(" line: ",i, line)
            sentence =" ".join(segmenter.segment(line.strip()))
            # print(" ".join(segmenter.segment(line.strip())))
            print(sentence)
            print(sentence[0],' ***** ', Pw[sentence[0]]/Pw.N)
            print('-'*60)
            if i ==1:
                break
            i += 1
