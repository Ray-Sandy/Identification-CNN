import urllib.request
import zipfile
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore")

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9 and logs.get('val_accuracy') > 0.9):
            print("\nReached or Exceed 98.0% accuracy so cancelling training!")
            self.model.stop_training = True

def identifikasi_daging():
    callback = myCallback()

    BASE_DIR = 'Dataset/Beef/Meat/'
    train_dir = os.path.join(BASE_DIR, 'train')
    # valid_dir = os.path.join(BASE_DIR, 'valid')

    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(312,312),
        color_mode='rgb',
        batch_size=16,
        class_mode='binary',
        subset='training',
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                     validation_split=0.1)
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=(312, 312),
        color_mode='rgb',
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )

    model = tf.keras.models.Sequential([
        # The first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(312, 312, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # dropout the results
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer activated by relu
        tf.keras.layers.Dense(512, activation='relu'),
        # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    val_step = val_generator.samples / val_generator.batch_size - 1
    model.fit(train_generator,
              validation_data=val_generator,
              # validation_steps=val_step,
              # steps_per_epoch=50,
              epochs=5,
              verbose=1,
              callbacks=callback,

    )

    return model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=identifikasi_daging()
    model.save("daging.h5")

