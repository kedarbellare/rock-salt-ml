import sys
stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, \
    Convolution2D, Dropout, Flatten
sys.stdout = stdout
sys.stderr = stderr
from sklearn.model_selection import ShuffleSplit
import numpy as np
import random

from learn.data import get_XY, ReplayFeaturizer
from utils.hlt import Move, Square, DIRECTIONS
from utils.replay import from_local, from_s3

nb_classes = 5
window = 7
default_input_shape = ((2 * window + 1) * 10,)
cnn_input_shape = (10, (2 * window + 1))


def get_linear_model(input_shape):
    return Sequential([
        Dense(nb_classes, input_shape=input_shape),
    ])


def get_mlp_model(input_shape):
    return Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(nb_classes),
    ])


def get_cnn_model(input_shape):
    return Sequential([
        Convolution2D(32, window + 1, window + 1, border_mode='valid',
                      dim_ordering='th', input_shape=input_shape),
        Activation('relu'),
        Dropout(0.25),
        Flatten(),
        Dense(16),
        Activation('relu'),
        Dropout(0.25),
        Dense(nb_classes),
    ])


def __default_model():
    base_model = get_mlp_model(default_input_shape)
    # base_model = get_cnn_model(cnn_input_shape)
    base_model.summary()
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
    return model


def save_model(model):
    print('>>> Saving model...', end='')
    model.save_weights('model.ser')
    with open('model.json', 'w') as fout:
        fout.write(model.to_json())
    print('Done.')


def load_model():
    with open('model.json') as fin:
        model = model_from_json(fin.read())
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.load_weights('model.ser')
    return model


def best_moves(model, frame, player):
    player_y, player_x = frame.player_yx(player)
    examples, _ = ReplayFeaturizer(player, window).examples_from_frame(frame)
    X = np.array(examples)
    X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    return [
        Move(
            Square(x, y, 0, 0, 0),
            DIRECTIONS[np.argmax(model.predict(np.array([vec]))[0])]
        )
        for x, y, vec in zip(player_x, player_y, X)
    ]


def learn_from_single_replay(replay):
    X, Y, w = get_XY(replay, player=replay.winner, window=window, linear=True)
    kf = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        w_train, w_test = w[train_index], w[test_index]

        print('#train:', X_train.shape)
        print('#test:', X_test.shape)

        model = __default_model()
        model.fit(X_train, Y_train, nb_epoch=30, verbose=1,
                  sample_weight=w_train,
                  validation_data=(X_test, Y_test, w_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        save_model(model)


def iter_data(fname, batch_size=100):
    with open(fname) as fp:
        replay_names = [line.rstrip('\r\n') for line in fp]
        random.shuffle(replay_names)
        for index, replay_name in enumerate(replay_names):
            try:
                replay = from_s3(replay_name)
                # replay = from_local(replay_name)
            except:
                continue

            print('Replay:', replay_name,
                  '(', (index + 1), '/', len(replay_names), ')',
                  'Winner:', replay.player_names[replay.winner - 1])
            X, Y, w = get_XY(replay, player=replay.winner, window=window,
                             linear=True)
            kf = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                w_train, w_test = w[train_index], w[test_index]

                print('#train:', X_train.shape)
                print('#test:', X_test.shape)

                for start in range(0, X_train.shape[0], batch_size):
                    begin, end = start, start + batch_size
                    yield X_train[begin:end], Y_train[begin:end], \
                        w_train[begin:end], True
                yield X_test, Y_test, w_test, False


def learn_from_replays(fname):
    model = __default_model()
    samples = 32
    num_samples_seen = 0
    checkpoint_samples = 1000000
    checkpoint_index = 0
    for epoch in range(10):
        print('>>> Epoch:', (epoch + 1))
        for X, Y, w, is_training in iter_data(fname, batch_size=samples):
            if is_training:
                model.train_on_batch(X, Y, sample_weight=w)
                num_samples_seen += X.shape[0]
                curr_checkpoint = int(num_samples_seen / checkpoint_samples)
                if curr_checkpoint > checkpoint_index:
                    checkpoint_index = curr_checkpoint
                    print('Processed:', num_samples_seen, 'samples')
                    save_model(model)
            else:
                score = model.evaluate(X, Y, verbose=0)
                print('Test:', score)


if __name__ == '__main__':
    learn_single = False
    if learn_single:
        learn_from_single_replay(from_local(sys.argv[1]))
    else:
        learn_from_replays(sys.argv[1])
