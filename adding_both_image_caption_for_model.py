from processing_images import *


#this will generate the data required and pad the sequence wherever needed



# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, train_features, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            real_path = 'rs\\Geetansh Kalra\\Desktop\\Internship 20jul-28Nov2020\\Breadth Projects\\Project1\\Images'
            real_key = os.path.join(real_path, key)
            print(real_key)
            n+=1
            # retrieve the photo feature
            photo = train_features[real_key+".jpg"]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0



#Like inceptionv3, glove is also pre-trained model for words
#Glove could give us required vectors for the words

# Load Glove vectors
glove_dir = 'Glove vectors'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))



embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector


print("Embedded matrix shape: ",embedding_matrix.shape)
print("--------------------------------------------------------------------------------------------------------------")

