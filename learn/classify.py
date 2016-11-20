from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import sys

from learn.data import get_XY
from utils.replay import from_local

nb_classes = 5


def get_linear_model(input_dim):
    return Sequential([
        Dense(nb_classes, input_dim=input_dim),
    ])


def get_mlp_model(input_dim):
    return Sequential([
        Dense(32, input_dim=input_dim),
        Activation('relu'),
        Dense(nb_classes),
    ])


if __name__ == '__main__':
    replay = from_local(sys.argv[1])
    X, Y = get_XY(replay, window=6, linear=True)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.4, random_state=0)

    print('#train:', X_train.shape[0])
    print('#test:', X_test.shape[0])

    base_model = get_mlp_model(X.shape[1])
    model = Sequential([
        base_model,
        Activation('softmax'),
    ])
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history = model.fit(X_train, Y_train, nb_epoch=20, verbose=1,
                        validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
