import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

# additional library
import operator

INDEX_PROBABILITY = 2
INDEX_WORD = 0
INDEX_STARTPOS = 1
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

        segmentation = iterative_segmentation(text,self.Pw,self.Pwords)
        # segmentation = iterative_segmentation(segmentation,self.Pw,self.Pwords)
        # list_keys = list(Pw.keys())
        # segmentation = recursive_segmentation(segmentation,list_keys)

        return segmentation

    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."
        # print([(self.Pw(w),w) for w in words], ' ]]]]]]]]]]]]]]]]]]]]]]]')

        return self.Pw(words)
        return product(self.Pw(w) for w in words)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)


# def recursive_segmentation(segmented_text, Pw):
#     if not segmented_text:
#         return ""
#     for i in range(len(segmented_text),-1,-1):
#         first_word = segmented_text[:i]
#         remainder = segmented_text[i:]
#         if first_word in Pw:
#             return first_word + " "+recursive_segmentation(remainder,Pw)
#     first_word = segmented_text[0]
#     remainder= segmented_text[1:]
#     return first_word+recursive_segmentation(remainder,Pw)

#### Support functions (p. 224)
def iterative_segmentation(text,Pw,Pwords):
    print('=============== ITERATIVE SEGMENTOR =================')

    def heappush_list(h, item, key=lambda x: x):
        heapq.heappush(h, (key(item), item))
    def heappop_list(h):
        return heapq.heappop(h)[1]

    '''Initialize the HEAP'''
    heap = []
    for key,value in dict(Pw).items():
        # if text[0] == key[0]:
        if (text[0] == key[0]) and len(key)==1:

            '''multiply by -1 to cast into positive
            then we can get Min Heap (minimum value at the top of heap) '''
            each_entry = [key,0,-1.0*log10(Pwords(key)),None]

            # each_entry = [key,1,-1.0*log10(Pwords(key)),None]

            heappush_list(heap, each_entry, key=operator.itemgetter(INDEX_PROBABILITY)) # sort by prob
            # heappush_list(heap, Entry(key,0,-1.0*log10(Pwords(key)),'blank'), key=operator.itemgetter(log_probability)) # sort by prob

    '''Iteratively fill in CHART for all i '''
    chart = {}
    count = 0
    while heap:
        print('WHILE: ',count)
        # print(chart)

        '''multiply by -1 to get original value back'''

        ''' get top entry from the heap'''
        entry = heappop_list(heap)
        ''' multiply -1 back to get original value of prob (original = negative log prob)'''
        entry[INDEX_PROBABILITY] = -1.0*entry[INDEX_PROBABILITY]
        print(entry, '<--- Entry')


        ''' Get the endindex-1 based on the length of the word in entry
        chart index is less than endindex in entry by -1
        note: endindex = length of word'''
        # endindex = len(entry[INDEX_WORD]) # index chart
        # endindex = count+len(entry[INDEX_WORD]) # index chart
        endindex = len(chart)
        # chartindex = endindex -1
        chartindex = endindex

        # print("endindex: ", endindex, " === chartindex: ",chartindex)
        print(heap[:5])
        for pword,value in dict(Pw).items():
            if len(chart) == len(text)-1:
                break
            if pword[0] == text[endindex+1]:

                if (pword in text):
                    new_entry = [pword, endindex + 1, -1.0 * (entry[INDEX_PROBABILITY] + log10(Pwords(pword))),
                                 entry[INDEX_STARTPOS]]
                    print(new_entry, log10(Pwords(pword)), " <-- New Entry")
                    # print(pword,value)

                    ''' don't add new word if it is equal to popped word'''
                    if pword == entry[INDEX_WORD]:
                        print('$$$$$$$$$$'*5)
                        continue
                    heappush_list(heap, new_entry, key=operator.itemgetter(INDEX_PROBABILITY))  # sort by prob


                    # print('heap: ',heap)
                    ## check if heap is empty, then add
                    # if not heap:
                    #     print('add to empty heap')
                    #     heappush_list(heap, new_entry, key=operator.itemgetter(INDEX_PROBABILITY))  # sort by prob
                    # #     print('heap 2: ',heap)
                    # else:
                    #     list_word_heap = []
                    #     for tuple_heap in heap:
                    #         list_word_heap.append(tuple_heap[1][INDEX_WORD])
                    #         if new_entry[INDEX_WORD] not in list_word_heap:
                    #             heappush_list(heap, new_entry, key=operator.itemgetter(INDEX_PROBABILITY))  # sort by prob


        def match_prev_entry(word_in_entry,chart):

            if chart[len(chart)-1][INDEX_WORD] == word_in_entry:
                return True
            # if chart[len(chart)-1][INDEX_WORD][-1] == word_in_entry:
                # return True
            return False
        # if chart and chart[entry[INDEX_STARTPOS]-2][INDEX_BACKPOINTER] != None:
        # if chart and chart[endindex-1][INDEX_BACKPOINTER] != None:
        if chart and match_prev_entry(entry[INDEX_WORD],chart):
            print('GO INSIDE IF-ELSE, has previous entry')
            print("@@@@@@@@@@@",chart)
            previous_entry = chart[chartindex-1]
            print("current prob: ", entry[INDEX_PROBABILITY]," -- previous prob: ", previous_entry[INDEX_PROBABILITY], ' #####')
            if entry[INDEX_PROBABILITY] > previous_entry[INDEX_PROBABILITY]:
                chart[chartindex] = entry
            if entry[INDEX_PROBABILITY] <= previous_entry[INDEX_PROBABILITY]:
                count += 1
                print('\n')
                continue

        else:
            print(" add to chart table !,: ",entry)
            chart[chartindex] = entry



        # print(heap)
        print(chart)
        print('-'*25,'\n')
        count += 1


    print('=============== END SEGMENTOR =================')
    print(chart)

    return get_segmented_text(chart)
    return [ w for w in text ]

def get_segmented_text(dict_text):
    ''' Get list of word from Dynamic programming table (chart) '''
    last_entry = dict_text[len(dict_text)-1]

    list_result = []

    # get last element
    list_result.append(last_entry[INDEX_WORD])
    # get pointer from last element
    ptr_idx = last_entry[INDEX_BACKPOINTER]

    # loop while backpoint is not None
    while ptr_idx != None:
        entry = dict_text[ptr_idx]
        list_result.append(entry[INDEX_WORD])
        ptr_idx = entry[INDEX_BACKPOINTER]

    #reverse list
    list_result = list_result[::-1]
    return list_result


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
            # if i == 1:
                # i += 1
                # continue
            print(" line: ",i, line)
            sentence =" ".join(segmenter.segment(line.strip()))
            # print(" ".join(segmenter.segment(line.strip())))
            print(sentence)
            # print(sentence[0],' ***** ', Pw[sentence[0]]/Pw.N)
            print('-'*60)
            if i ==1:
                break
            i += 1
