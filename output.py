from final_model import *
from adding_both_image_caption_for_model import *

model.load_weights('final_model_weights.h5')

images = 'Images'


with open("final_encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)


def search(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

z = random.randint(0,2091)
print("The index value is: ", z)
pic_real_path = "C:\\Users\\Geetansh Kalra\\Desktop\\Internship 20jul-28Nov2020\\Breath projects\\Project1\\Images\\"
pic = list(encoding_test.keys())[z]
image_to_show = pic_real_path+test[z]+".jpg"
image = encoding_test[pic].reshape((1,2048))

x=plt.imread(image_to_show)



print("----------------------------------------------------------------------------------------------------------")
print("Caption for the Image: ",search(image))
print("-----------------------------------------------------------------------------------------------------------")
plt.imshow(x)
plt.show()
