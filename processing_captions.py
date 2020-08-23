from dividing_dataset import *
import pickle
import numpy as np



from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model

from keras.applications.inception_v3 import preprocess_input





#train : variable in which all images name are stored in form of list
#test


# a start sequence and end sequence must be added in all captions in start and in the end respectively
# this is done so that predicting model knows when to start and when to stop writing a caption for the target image

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()   # split line by white space

        image_id, image_desc = tokens[0], tokens[1:]  # split id from description

        if image_id in dataset:   # skip images not in the train list

            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            descriptions[image_id].append(desc)

    return descriptions


# loading our saved descriptions file and our train list which contains all the image_id
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Made a new train_descriptions dict which contains image id and captions of that image with "startseq" and "endseq"')
print("For example: ", list(train_descriptions.items())[:2])
print("--------------------------------------------------------------------------------------------------------------")



# Create a list of all the training captions
#our training captions our stored in an unhashable list, so we will convert them into a normal list to get access to each caption

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print("All caption stored in a list")
print("Length of all training captions list",len(all_train_captions)) #this should be 30000 as we have 6000 train images and 5 captions for each image


# we have around 8680 words in our vocabulary, but many of them might not occur frequently
# so we have to focus on those words which occur more frequently, let's keep a threshold count of 10
# so we will consider only those words in our vocabulary which occur more than 10 times
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words/ change of our vocabulary from  %d -> %d' % (len(word_counts), len(vocab)))




# NN can't take strings as an input, so we convert each word into an integer
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix # assigning each word an integer
    ixtoword[ix] = w # this is to cross verify
    ix += 1


vocab_size = len(ixtoword) + 1 # one for appended 0's
print("New Vocabulary Size", vocab_size)
print("-----------------------------------------------------------------------------------------------------")





# calculating max. length of each descriptions, This we our doing so that we can add padding to a certain length

def to_lines(train_descriptions):
    all_desc = list()
    for key in train_descriptions.keys():
        for d in train_descriptions[key]:
            all_desc.append(d)
    return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Max Caption Length is : %d' % max_length)
print("--------------------------------------------------------------------------------------------------------------")





