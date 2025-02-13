# Code adapted from original code by Robert Guthrie
import os, sys, optparse, gzip, re, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

import string
import numpy as np

INDEX_BEG = 0
LENGTH_VECTOR_ENCODING = 300
# dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

def prepare_sequence(seq, to_ix, unk):
    idxs = []
    if unk not in to_ix:
        idxs = [to_ix[w] for w in seq]
    else:
        idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
    return torch.tensor(idxs, dtype=torch.long)

################################################################
'''Helper functions'''

def encode_one_char(sentence,vector_2d,first_char=True):
    ''' vector size = string.printable == 100
     one-hot encoding only the first/last letter of word '''

    '''if word exists in string.printable, use 1; otherwise, 0'''
    # iterate word by word in the sentence
    for word,vector in zip(sentence,vector_2d):
        if word == '[UNK]':
            continue

        if first_char:
            index_word = INDEX_BEG
        else:
            index_word = len(word)-1

        letter = word[index_word]
        index_strPrintable= string.printable.find(letter) # letter to index
        vector[index_strPrintable] = 1.0
    return vector_2d

def encode_internal_chars(sentence,vector_2d):
    ''' vector size = string.printable == 100
     Encoding only the internal letters of word  (excluding begining and ending chars)'''

    # iterate word by word in the sentence
    for word,vector in zip(sentence,vector_2d):
        if word == '[UNK]':
            continue

        # if word length < 3, there is no internal
        if len(word) < 3:
            continue

        internal_word = word[1:-1]
        # interate letter in the internal word
        for letter in internal_word:
            index_strPrintable= string.printable.find(letter) # letter to index
            vector[index_strPrintable] += 1.0
    return vector_2d

def encoding_sentence(sentence):
    '''Function to encode every word in the sentence'''

    '''encoding the beginning charactor of all words in the sentence'''
    beginChar_vector = np.zeros((len(sentence),len(string.printable)))
    beginChar_vector = encode_one_char(sentence,beginChar_vector,first_char=True)

    '''encoding the ending charactor of all words in the sentence'''
    endChar_vector = np.zeros((len(sentence),len(string.printable)))
    endChar_vector = encode_one_char(sentence,endChar_vector,first_char=False)

    '''encoding all internal charactors of all words in the sentence'''
    internal_vector = np.zeros((len(sentence),len(string.printable)))
    internal_vector = encode_internal_chars(sentence,internal_vector)

    ''' concate all 3 vectors '''
    encoding_vector = np.concatenate((beginChar_vector,internal_vector,endChar_vector),axis=1)
    # print(beginChar_vector.shape,endChar_vector.shape,internal_vector.shape,encoding_vector.shape)

    '''create Tensor from numpy object '''
    # encoding_tensor = torch.tensor(encoding_vector, dtype=torch.float)
    encoding_tensor = torch.tensor(encoding_vector, dtype=torch.float).cuda()

    return encoding_tensor
############################################################


class LSTMTaggerModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,char_encoding):
        torch.manual_seed(1)
        super(LSTMTaggerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        '''if using character-level encoding, hidden dim to lstm = 128+300 = 428'''
        if char_encoding:
            lstm_embedding_dim = embedding_dim+LENGTH_VECTOR_ENCODING
        else:
            lstm_embedding_dim = embedding_dim

        self.lstm = nn.LSTM(lstm_embedding_dim, hidden_dim, bidirectional=False)

        ## second LSTM for character-level encoding vector
        # self.lstm_encoding = nn.LSTM(LENGTH_VECTOR_ENCODING, hidden_dim, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence,encoding_tensor=None):
        embeds = self.word_embeddings(sentence)

        # # put character-level encoding vectors into LSTM before concatenating with embedding
        # if encoding_tensor is not None:
        #     reshaped_encoding_tensor = torch.reshape(encoding_tensor,(-1,1,LENGTH_VECTOR_ENCODING))
        #     lstm_out_encoding, _ = self.lstm_encoding(reshaped_encoding_tensor)
        #     encoding_tensor = torch.reshape(lstm_out_encoding,(-1,LENGTH_VECTOR_ENCODING))

        '''if using character-level encoding, we concatenate Embedding vector with new encoded vectors = 128+300 = 428'''
        if encoding_tensor is not None:
            embeds = torch.cat([embeds,encoding_tensor],dim=1)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class LSTMTagger:

    def __init__(self, trainfile, modelfile, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64,char_encoding=False):
        self.unk = unk
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.training_data = []
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        self.word_to_ix = {} # replaces words with an index (one-hot vector)
        self.tag_to_ix = {} # replace output labels / tags with an index
        self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag

        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        logging.info("word_to_ix:", self.word_to_ix)
        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)

        '''Flag whether do character-level encoding or not'''
        self.char_encoding = char_encoding
        # self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix))
        self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix),char_encoding=char_encoding).cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def argmax(self, seq):
        output = []
        with torch.no_grad():
            inputs = prepare_sequence(seq, self.word_to_ix, self.unk)

            '''if using character-level encoding, we encode the tensor'''
            if self.char_encoding:
                encoding_tensor = encoding_sentence(seq)
            else:
                encoding_tensor= None

            tag_scores = self.model(inputs,encoding_tensor)
            for i in range(len(inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output


    def train(self):
        loss_function = nn.NLLLoss()

        self.model.train()
        loss = float("inf")
        i = 0
        for epoch in range(self.epochs):
            # if i == 4:
                # break
            for sentence, tags in tqdm.tqdm(self.training_data):
                i+= 1
                # if i == 4:
                    # break

                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                # sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
                sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk).cuda()

                # targets = prepare_sequence(tags, self.tag_to_ix, self.unk)
                targets = prepare_sequence(tags, self.tag_to_ix, self.unk).cuda()

                # Step 3. Run our forward pass.
                # tag_scores = self.model(sentence_in)
                '''if using character-level encoding, we encode the tensor'''
                # create character level vectors to concate with the embeddings
                if self.char_encoding:
                    encoding_tensor = encoding_sentence(sentence)
                else:
                    encoding_tensor = None

                tag_scores = self.model(sentence_in,encoding_tensor=encoding_tensor)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                self.optimizer.step()

            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        loss = saved_model['loss']
        self.unk = saved_model['unk']
        self.word_to_ix = saved_model['word_to_ix']
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        self.model.eval()
        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax(sent))
        return decoder_output

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
    optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'chunker'), help="filename without suffix for model files")
    optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
    optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
    optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-o", "--outputfile", dest="outputfile", default='output.txt', help="print result to output file")

    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    if opts.modelfile[-4:] == '.tar':
        modelfile = opts.modelfile[:-4]
    # chunker = LSTMTagger(opts.trainfile, modelfile, opts.modelsuffix, opts.unk,char_encoding=False)
    chunker = LSTMTagger(opts.trainfile, modelfile, opts.modelsuffix, opts.unk,char_encoding=True)

    # use the model file if available and opts.force is False
    if os.path.isfile(opts.modelfile + opts.modelsuffix) and not opts.force:
        decoder_output = chunker.decode(opts.inputfile)
    else:
        print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
        chunker.train()
        decoder_output = chunker.decode(opts.inputfile)

    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))

    ''' Print out to file instead of using python ... > ... '''
    original_stdout = sys.stdout
    with open(opts.outputfile, 'w') as f:

        sys.stdout = f # Change the standard output to the file we created.
        print("\n\n".join([ "\n".join(output) for output in decoder_output ]),flush=True)
        sys.stdout = original_stdout
