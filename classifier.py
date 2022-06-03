from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
#import sequential model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
#from #keras.models import model_from_json


data=[]
labels=[]
cats=os.listdir('C:/Users/hp user/Desktop/Python/image_classifier/train/cats')

for cat in cats:
 imag=cv2.imread("C:/Users/hp user/Desktop/Python/image_classifier/train/cats/"+cat,1)
 img_from_ar = Image.fromarray(imag, 'RGB')
 resized_image = img_from_ar.resize((50, 50))
 data.append(np.array(resized_image))
 labels.append(0)

dogs=os.listdir("C:/Users/hp user/Desktop/Python/image_classifier/train/dogs")
for dog in dogs:
 imag=cv2.imread("C:/Users/hp user/Desktop/Python/image_classifier/train/dogs/"+dog,1)
 img_from_ar = Image.fromarray(imag, 'RGB')
 resized_image = img_from_ar.resize((50, 50))
 data.append(np.array(resized_image))
 labels.append(1)

 pigs=os.listdir("C:/Users/hp user/Desktop/Python/image_classifier/train/pigs")
for pig in pigs:
 imag=cv2.imread("C:/Users/hp user/Desktop/Python/image_classifier/train/pigs/"+pig,1)
 img_from_ar = Image.fromarray(imag, 'RGB')
 resized_image = img_from_ar.resize((50, 50))
 data.append(np.array(resized_image))
 labels.append(2)

 lions=os.listdir("C:/Users/hp user/Desktop/Python/image_classifier/train/lions")
for lion in lions:
 imag=cv2.imread("C:/Users/hp user/Desktop/Python/image_classifier/train/lions/"+lion,1)
 img_from_ar = Image.fromarray(imag, 'RGB')
 resized_image = img_from_ar.resize((50, 50))
 data.append(np.array(resized_image))
 labels.append(3)

 zebras=os.listdir("C:/Users/hp user/Desktop/Python/image_classifier/train/zebras")
for zebra in zebras:
 imag=cv2.imread("C:/Users/hp user/Desktop/Python/image_classifier/train/zebras/"+zebra,1)
 img_from_ar = Image.fromarray(imag, 'RGB')
 resized_image = img_from_ar.resize((50, 50))
 data.append(np.array(resized_image))
 labels.append(4)

#  goats=os.listdir("C:/Users/hp user/Desktop/Python/image_classifier/train/goats")
# for goat in goats:
#  imag=cv2.imread("C:/Users/hp user/Desktop/Python/image_classifier/train/goats/"+goat,1)
#  img_from_ar = Image.fromarray(imag, 'RGB')
#  resized_image = img_from_ar.resize((50, 50))
#  data.append(np.array(resized_image))
#  labels.append(5)

#converting to numpy arrays
animals=np.array(data)

labels=np.array(labels)

#saving the arrays
np.save("animals", animals)

np.save("labels", labels)

#lload the arrays
animals= np.load("animals.npy")
labels=np.load("labels.npy")
#shuffling the dataset
s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

# variablle numclasses
num_classes=len(np.unique(labels))
data_length=len(animals)

#dividing data into test and train sets
(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

#divide labels into test and train
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

#One hot encoding
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)
y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3))),model.add(MaxPooling2D(pool_size=2)),model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu")),model.add(MaxPooling2D(pool_size=2)),model.add(Conv2D(filters=64,kernel_size=2,padding="same",
activation="relu")),model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5,activation="softmax"))
model.summary()

# compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50,epochs=100,verbose=1)
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

def convert_to_array(img):
    im = cv2.imread(img )
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)


def get_animal_name(label):
    if label == 0:
        return "cat"
    if label == 1:
        return "dog"
    if label == 2:
      return "pig"
    if label == 3:
      return "lion"
    if label == 4:
      return "zebra"
    # if label == 5:
    #   return "goat"


#predict animal
def predict_animal(file):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar / 255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a, verbose=1)
    print(score)
    label_index = np.argmax(score)
    print(label_index)
    acc = np.max(score)
    animal = get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a " + str(animal) + " with accuracy = " + str(acc))
    
predict_animal ("C:/Users/hp user/Desktop/Python/image_classifier/train/lions/lion01.jpeg")

model.save('model.h5')

