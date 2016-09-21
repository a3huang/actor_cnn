from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import Image
import numpy as np
import os, h5py, cv2

# put own path here
train_directory = ''
val_directory = ''

nbt = [len(files) for r, d, files in sorted(os.walk(train_directory))][1:]
nbv = [len(files) for r, d, files in sorted(os.walk(val_directory))][1:]
num_cat = len(nbv)
total_t = sum(nbt)
total_v = sum(nbv)

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
	
    # load in weights for convolutional layers
    if weights_path:
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):			
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] \
				for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
    datagen.mean = np.array([103.939, 116.779, 123.68],
        dtype=np.float32).reshape(3,1,1)
    
    # pass in training data
    generator = datagen.flow_from_directory(
        train_directory,
        target_size=(224, 224),
        batch_size=32,
        classes=None,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, total_t)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    print 'Training data saved.'
    
    # pass in validation data
    generator = datagen.flow_from_directory(
        val_directory,
        target_size=(224, 224),
        batch_size=32,
        classes=None,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, total_v)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    print 'Validation data saved.'

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    labels = np.array(np.sum([[i]*nbt[i] for i in range(num_cat)]))
    labels = to_categorical(labels, num_cat)
    train_labels = np.array(labels)
        
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    labels = np.array(np.sum([[i]*nbv[i] for i in range(num_cat)]))
    labels = to_categorical(labels, num_cat)
    validation_labels = np.array(labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_cat, activation='softmax'))
    
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=50, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights('fc_model.h5')

if __name__ == "__main__":
    # Train model on previous weights and movie data
    VGG_16('vgg16_weights.h5')
    train_top_model()
