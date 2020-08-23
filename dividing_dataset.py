from cleaning_captions import *
import os

print("Dividing our dataset into train and test set")

#below train and test lists contains names of all the images
train = []
for i in list(descriptions.keys()):
    train.append(i)

train = train[:6000] #will have 6000 images in train

#rest of the images in test set
test = []
for i in range(6000,len(list(descriptions.keys()))):
    test.append(list(descriptions.keys())[i])



print("Length of test set:" , len(test))
print('Length of train set: %d' % len(train))
print("------------------------------------------------------------------------------------------------------------")



# to access our sets {train and test images} we have to make a variable which could store images paths.

folder = "Images"  #folder in the current directory which contains all the images
train_img_path = []
path = os.getcwd()
new_path = os.path.join(path, folder)
for i in train:
    i = i + ".jpg"
    i = os.path.join(new_path, i)
    train_img_path.append(i)

train_img = train_img_path[:] # contains all train images

print("All train images path loaded")
print("for eg:", train_img[:2])

test_img_path = []
path = os.getcwd()
new_path = os.path.join(path, folder)
for i in test:
    i = i + ".jpg"
    i = os.path.join(new_path, i)
    test_img_path.append(i)
test_img = test_img_path[:] # contains all test images
print("All test images path loaded")
print("for eg: ", test_img[:2])

print("-----------------------------------------------------------------------------------------------------------")
