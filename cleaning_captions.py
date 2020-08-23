import string

from loading_captions_images import *

print("----------------------------------------------------------------------------------------------------------")
print("Cleaning the descriptions")

def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)


# clean descriptions
clean_descriptions(descriptions)

print("All captions cleaned")
print("-------------------------------------------------------------------------------------------------------------")


# save descriptions to file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


save_descriptions(descriptions, 'descriptions.txt')
print("Descriptions have been saved to a file named descriptions.txt")
print("-------------------------------------------------------------------------------------------------------------")





#Vocabulary
#Let's see some unique words in the captions and make vocabulary of those words
def to_vocabulary(descriptions):
    all_desc = set() # make a set so that no words gets repeated
    for key in descriptions.keys():
        for d in descriptions[key]:
            d = d.split()
            all_desc.update(d)
    return all_desc


vocabulary = to_vocabulary(descriptions)
print('Original Vocabulary Size: %d' % len(vocabulary))
print("------------------------------------------------------------------------------------------------------------")


