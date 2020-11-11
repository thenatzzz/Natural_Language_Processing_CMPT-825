# from datasets import load_dataset
#
# dataset =load_dataset('cnn_dailymail','3.0.0')
# print(len(dataset))
# print(dataset)
# print()


from tqdm import tqdm
from os import listdir
import string
# from pickle import dump
import pickle

# load doc into memory


def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights


def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories in a directory


def load_stories(directory):
    stories = list()
    for name in tqdm(listdir(directory)):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories

# clean a list of lines


def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # print(']]]]] ', line)
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


if __name__ == '__main__':
    # load stories
    directory = 'data_zip/cnn/stories/'
    # directory = 'data_zip/cnn/samples/'

    # stories = load_stories(directory)
    # print('Loaded Stories %d' % len(stories))
    #
    # print('-'*40)
    # print(stories[1]['story'])
    # print('-'*40)
    # print(stories[1]['highlights'])
    #
    # print('='*40)
    # example = {}
    # example['story'] = clean_lines(stories[1]['story'].split('\n'))
    # example['highlights'] = clean_lines(stories[1]['highlights'])
    # print(example['story'])
    # print('-'*40)
    # print(example['highlights'])
    ## clean stories
    # for example in stories:
    	# example['story'] = clean_lines(example['story'].split('\n'))
    	# example['highlights'] = clean_lines(example['highlights'])
    # print(stories[1])
    # pickle.dump(stories, open('cnn_dataset.pkl', 'wb'))

    # load from file
    stories = pickle.load(open('data/processedID_cnn_dataset.pkl', 'rb'))
    print('Loaded Stories %d' % len(stories))
    print(stories[23944])
    # i = 0
    # for example in stories:
        # example['id'] = 'id_'+str(i)
        # i += 1
    # pickle.dump(stories, open('processedID_cnn_dataset.pkl', 'wb'))
