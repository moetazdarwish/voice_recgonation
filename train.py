import json
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
DATA_PATH = "data.json"
LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 32
SAVED_MODEL_PATH = "../flask/model.h5"
NUM_KEYWORDS = 10
def load_dataset(data_path):
    with open(data_path , "r") as fp:
        data = json.load(fp)
    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return x ,y


def get_data_splits(data_path, TEST_SIZE=0.1, TEST_VALIDATION= 0.1):
    x,y = load_dataset(data_path)

    X_train , X_test , Y_train, Y_test = train_test_split(x, y, test_size=TEST_SIZE)
    X_train , X_validation , Y_train, Y_validation = train_test_split(X_train, Y_train,
                                                                      test_size=TEST_VALIDATION)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test= X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64 , (3,3),activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2), padding="same"))

    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                           input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                           input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))



    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(NUM_KEYWORDS , activation="softmax"))


    optimiser =keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer=optimiser, loss=error , metrics=["accuracy"])


    
    model.summary()
    return  model




def main():

    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = get_data_splits(DATA_PATH)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape,LEARNING_RATE)

    # train
    model.fit(X_train, Y_train, epochs=EPOCHS , batch_size=BATCH_SIZE, validation_data=(X_validation,Y_validation))

    test_error , test_accuracy = model.evaluate(X_test, Y_test)

    print(f"test error :{test_error},test_accuracy: {test_accuracy}")

    model.save(SAVED_MODEL_PATH)

if __name__ =="__main__":
    main()