from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D


def load(csv):
    csv_file = csv
    dataframe = pd.read_csv(csv_file,
                            usecols=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour',
                                     'season', 'year', 'usage', 'productDisplayName'])

    return dataframe


def dataPreparation(csv, data):
    csv['id'] = csv['id'].fillna(0)

    csv['gender'] = csv['gender'].fillna(0)
    csv['masterCategory'] = csv['masterCategory'].fillna(0)
    csv['subCategory'] = csv['subCategory'].fillna(0)
    csv['articleType'] = csv['articleType'].fillna(0)
    csv['baseColour'] = csv['baseColour'].fillna(0)
    csv['season'] = csv['season'].fillna(0)
    csv['year'] = csv['year'].fillna(0)
    csv['usage'] = csv['usage'].fillna(0)
    csv['productDisplayName'] = csv['productDisplayName'].fillna(0)

    CounterDat = 0
    Counter = 0

    # kontrola parametrov
    for index, row in csv.iterrows():
        if (row['id'] != 0 and row['gender'] != 0 and row['masterCategory'] != 0 and row['subCategory'] != 0
                and row['articleType'] != 0 and row['baseColour'] != 0 and row['season'] != 0
                and row['year'] != 0 and row['usage'] != 0 and row['productDisplayName'] != 0
                and row['masterCategory'] != "home" and row['masterCategory'] != "sporting-goods" and row[
                    'masterCategory'] != "free-items"):
            data.loc[CounterDat] = [csv.loc[Counter, 'id']] + [csv.loc[Counter, 'gender']] + [
                csv.loc[Counter, 'masterCategory']] + [csv.loc[Counter, 'subCategory']] + [
                                      csv.loc[Counter, 'articleType']] + [csv.loc[Counter, 'baseColour']] + [
                                      csv.loc[Counter, 'season']] + [csv.loc[Counter, 'year']] + [
                                      csv.loc[Counter, 'usage']] + [csv.loc[Counter, 'productDisplayName']]

            CounterDat = CounterDat + 1

        Counter = Counter + 1

    data.to_csv("styles_fixed.csv")  # spravime nove .csv pre testovanie

    data.drop(data[data['id'] == 39403].index, inplace=True)
    data.drop(data[data['id'] == 39410].index, inplace=True)
    data.drop(data[data['id'] == 39425].index, inplace=True)
    data.drop(data[data['id'] == 39401].index, inplace=True)
    data.drop(data[data['id'] == 12347].index, inplace=True)

    return data

def addFolders():
    import os
    path0 = os.getcwd()

    path = path0 + "/Data/Accessories"
    path2 = path0 + "/Data/Apparel"
    path3 = path0 + "/Data/Footwear"
    path4 = path0 + "/Data/Free Items"
    path5 = path0 + "/Data/Home"
    path6 = path0 + "/Data/Personal Care"
    path7 = path0 + "/Data/Sporting Goods"

    try:
        os.makedirs(path)
        os.makedirs(path2)
        os.makedirs(path3)
        os.makedirs(path4)
        os.makedirs(path5)
        os.makedirs(path6)
        os.makedirs(path7)
    except OSError:
        print("Creation of the directory %s failed" % path)
        print("Creation of the directory %s failed" % path2)
        print("Creation of the directory %s failed" % path3)
        print("Creation of the directory %s failed" % path4)
        print("Creation of the directory %s failed" % path5)
        print("Creation of the directory %s failed" % path6)
        print("Creation of the directory %s failed" % path7)
    else:
        print("Successfully created the directory %s" % path)
        print("Successfully created the directory %s" % path2)
        print("Successfully created the directory %s" % path3)
        print("Successfully created the directory %s" % path4)
        print("Successfully created the directory %s" % path5)
        print("Successfully created the directory %s" % path6)
        print("Successfully created the directory %s" % path7)


def imageLoad():
    df = pd.read_csv("styles_fixed.csv")

    for index, row in df.iterrows():

        if (row['id'] != 39403 and row['id'] != 12347 and row['id'] != 39401 and row['id'] != 39410 and row['id'] != 39425):
            datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, rescale=1. / 255,
                                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
            img = load_img('images/' + str(row['id']) + '.jpg')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            x.shape

            i = 0
            for j in datagen.flow(x, batch_size=1, save_to_dir='Data/' + row['masterCategory'],
                                      save_prefix=str(row['id']) + '_' + row['masterCategory'], save_format='jpeg'):
                i += 1
                if i > 1:
                    break


df = pd.DataFrame(columns=['id','gender','masterCategory','subCategory','articleType','baseColour','season','year','usage','productDisplayName'])
df2 = pd.DataFrame(columns=['id','gender','masterCategory','subCategory','articleType','baseColour','season','year','usage','productDisplayName'])

df = load("styles.csv")
print(df)

df = dataPreparation(df,df2)


addFolders()
imageLoad()

#Convolution network

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation="softmax"))

optimizer = optimizers.Adam(learning_rate=0.001) #0.001 , 0.0001 best LRate

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

batch_size = 16
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    
train_dataset = image_generator.flow_from_directory(batch_size=32, directory='Data', shuffle=True, target_size=(150, 150), 
                                                    subset="training", class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32, directory='Data', shuffle=True, 
                                                         target_size=(150, 150), subset="validation", class_mode='categorical')

history = model.fit_generator(train_dataset, steps_per_epoch=2000 // batch_size, epochs=20, 
                               validation_data=validation_dataset, validation_steps=800 // batch_size)

print("History keys : ",history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','valid'],loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
