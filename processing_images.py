from processing_captions import *

# We will be using pre-trained inception v3 model here and extract features from it for our images


'''
# Load the inception v3 model
model = InceptionV3(weights='imagenet')
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)




# Already encoded the images we have
# don't run this again
#this takes time

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


images = "Images"
# Call the function to encode all the train images
# This will take a while
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)



with open("final_encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)


encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)

with open("final_encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)


'''


# all the features that our model has learned were stored in train_features
# which contain image id and features
train_features = load(open("final_encoded_train_images.pkl", "rb"))
print('Trained features Length : train=%d' % len(train_features))
print("For example: ",list(train_features.items())[:2])

