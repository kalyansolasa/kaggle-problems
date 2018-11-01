import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2 as cv
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
folder_path = "<folder path>"
print(os.listdir(folder_path))
df = pd.read_csv(folder_path+'/sample_submission.csv')
train_dir_path = folder_path+"/train"
test_dir_path = folder_path+"/test"
#pickled_dir_path  = "../output/pickled_Data"
labels_df = pd.read_csv(folder_path+'/labels.csv')
dog_breeds = list(df.columns[1:])
print(len(dog_breeds))
print(dog_breeds)
# variables 
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_CHANNELS = 3
BATCH_SIZE = 500
def img_to_array(img_path):   
    img_array = cv.imread(img_path)
    img_array = cv.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_array.reshape(-1,IMG_HEIGHT,IMG_WIDTH, IMG_CHANNELS)
    return img_array
def dog_breed_from_id(dog_id):
    #labels_df = pd.read_csv('../input/labels.csv')
    return labels_df[labels_df['id'] == dog_id]['breed'].values

# method for onehot encoding labels of train_arr
def one_hot_encode_labels(label_arr):
    labelEncoder = LabelEncoder()
    integer_encoded = labelEncoder.fit_transform(np.array(label_arr))
    integer_encoded = integer_encoded.reshape(-1,1)
    onehotEncoder = OneHotEncoder()
    onehot_encoded_arr = onehotEncoder.fit_transform(integer_encoded).toarray()
    return onehot_encoded_arr


def get_train_data():
    train_path = os.path.join(folder_path,'train')
    results = []
    img_arrays = []
    train_labels = []
    count = 0
    for f_path in os.listdir(train_path):
        img_id = os.path.basename(f_path).split('.')[0]
        img_arrays.append(img_to_array(os.path.join(train_dir_path,f_path)))
        train_labels.append(dog_breed_from_id(img_id))
    train_arr = np.array(img_arrays).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    train_arr = train_arr/255
    train_labels = one_hot_encode_labels(train_labels)
    
    return train_arr, train_labels





x, y =get_train_data()


train_x, valdn_x, train_y, valdn_y = train_test_split(x,y,test_size=0.3)
print('train data done')

def get_test_data():
    test_fol_path = os.path.join(folder_path, 'test')
    test_img_ids = []
    test_img_list = []
    for f_path in os.listdir(test_dir_path):
        img_id = os.path.basename(os.path.join(test_fol_path,f_path)).split('.')[0]
        test_img_list.append(img_to_array(os.path.join(test_fol_path,f_path)))
        test_img_ids.append(img_id)
    test_img_arr = np.array(test_img_list).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    test_img_arr = test_img_arr/255
    return test_img_arr, test_img_ids    
    

test_x, test_img_ids = get_test_data()
print('test data done')



# CNN model
model = Sequential()

# -----------------------------------------------------------------------------------
# conv 1
model.add(Conv2D(16, (3,3), input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))       # input -N,150,150,3, output- N,148,148,16
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# max pool 1
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                                   #input- N,148,148,16, output- N, 74,74,16

# -----------------------------------------------------------------------------------
# # conv 2
model.add(Conv2D(32, (3,3)))                                                         #input- N,74,74,16 output - N, 72,72,16
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# max pool 2
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                                 #input - N,72,72,16, output- N,36,36,16
# -----------------------------------------------------------------------------------

# conv 3
model.add(Conv2D(48, (3,3)))                                                       #input - N,36,36,16, output- N,34,34,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.7))

# max pool 3
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                                #input- N,34,34,32, output- N,17,17,32
# -----------------------------------------------------------------------------------

# # conv 4
model.add(Conv2D(64, (3,3)))                                                     #input- N,17,17,32, output- N,15,15,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.7))
# max pool 4
model.add(MaxPooling2D(pool_size=(2,2),strides=2))                              #input- N,15,15,32, output- N,7,7,32

# flatten
model.add(Flatten())                                                            # output- 1568

# fc layer 1
model.add(Dense(1024, activation='relu'))                                  

# fc layer 2
model.add(Dense(512, activation='relu'))

# fc layer 3
model.add(Dense(256, activation='relu'))

# fc layer 4
model.add(Dense(120, activation='softmax'))



model.summary()


# compile model for with softmax cross entropy and adam optimizer, set accuracy as parameter to evaluate
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#early stoping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

# train model on training data
model_hist = model.fit(train_x, train_y, batch_size=64, nb_epoch=100, verbose=1, validation_data=(valdn_x, valdn_y), callbacks=[early_stopping])


predictions = model.predict(test_x, batch_size=32, verbose=1)



print(predictions.shape)
print(len(dog_breeds))



import pandas as pd
submission_res = pd.DataFrame(data= predictions, index =test_img_ids, columns= dog_breeds)
submission_res.index.name = 'id'
submission_res.to_csv('submission.csv', encoding='utf-8', index=True)



