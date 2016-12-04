import numpy as np

import plac
import random
import ujson as json

from learn.keras_utils import np_utils, models, ModelCheckpoint, Sequential
from learn.keras_utils import Activation, Convolution1D, Convolution2D, \
    Dense, Dropout, Flatten
from learn.features import \
    process_frame_axes, process_frame_tile, \
    process_replay
from sklearn.model_selection import ShuffleSplit
from utils.hlt import Move, Square, DIRECTIONS
from utils.replay import from_local, from_s3

nb_classes = len(DIRECTIONS)
FEATURE_TYPE_PROCESSOR = {
    'axes': process_frame_axes,
    'tile': process_frame_tile,
}


def get_linear_model(input_shape):
    return Sequential([
        Dense(nb_classes, input_shape=input_shape),
    ])


def get_mlp_model(input_shape):
    return Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dropout(0.25),
        Dense(256, activation='relu'),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(nb_classes),
    ])


def get_cnn2d_model(input_shape):
    """
    Only compatible with 2-dimensional tile-based features
    """
    return Sequential([
        Convolution2D(32, 5, 5,
                      border_mode='valid', dim_ordering='th',
                      activation='relu', input_shape=input_shape),
        Dropout(0.25),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.25),
        Dense(nb_classes),
    ])


def get_cnn1d_model(input_shape):
    """
    Only compatible with 1-dimensional axis-based features
    """
    return Sequential([
        Convolution1D(256, 3, border_mode='valid', activation='relu',
                      input_shape=input_shape),
        Dropout(0.25),
        Convolution1D(128, 3, border_mode='valid', activation='relu'),
        Dropout(0.1),
        Convolution1D(64, 3, border_mode='valid', activation='relu'),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(nb_classes),
    ])


def create_model(X, **learn_args):
    input_shape = X.shape[1:]
    model_type = learn_args['model_type']
    if model_type == 'cnn2d':
        base_model = get_cnn2d_model(input_shape)
    elif model_type == 'cnn1d':
        base_model = get_cnn1d_model(input_shape)
    elif model_type == 'mlp':
        base_model = get_mlp_model(input_shape)
    elif model_type == 'linear':
        base_model = get_linear_model(input_shape)
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


def save_model(model, **learn_args):
    print('>>> Saving model...')
    model.save('%s.hd5' % learn_args['model_prefix'])
    print('Done.')


def load_model(**learn_args):
    return models.load_model('%s.hd5' % learn_args['model_prefix'])


def best_moves(model, frame, player, **learn_args):
    player_y, player_x = frame.player_yx(player)
    frame_processor = FEATURE_TYPE_PROCESSOR[learn_args['feature_type']]
    examples, _ = frame_processor(frame, player, window=learn_args['window'])
    X = np.array(examples)
    if learn_args['linear']:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    return [
        Move(
            Square(x, y, 0, 0, 0),
            DIRECTIONS[np.argmax(model.predict(np.array([vec]))[0])]
        )
        for x, y, vec in zip(player_x, player_y, X)
    ]


def get_XY(replay, **learn_args):
    # construct input and output
    examples, labels = process_replay(
        FEATURE_TYPE_PROCESSOR[learn_args['feature_type']],
        replay,
        player=replay.winner,
        window=learn_args['window']
    )
    X, y = np.array(examples), np.array(labels, dtype=int)
    Y = np_utils.to_categorical(y, nb_classes=nb_classes)
    if learn_args['linear']:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    return X, Y


def get_train_test_data(replay, **learn_args):
    X, Y = get_XY(replay, **learn_args)

    # create splits for train/test
    kf = ShuffleSplit(
        n_splits=1,
        test_size=learn_args['test_size'],
        random_state=0
    )
    train_index, test_index = next(kf.split(X))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    print('#train:', X_train.shape)
    print('#test:', X_test.shape)
    return X_train, Y_train, X_test, Y_test


def learn_from_single_replay(input_file, **learn_args):
    replay = from_local(input_file) if learn_args['local_replays'] else \
        from_s3(input_file)

    X_train, Y_train, X_test, Y_test = \
        get_train_test_data(replay, **learn_args)

    # create model
    model = create_model(X_train, **learn_args)
    model.fit(X_train, Y_train, nb_epoch=20, verbose=1,
              callbacks=[ModelCheckpoint(
                  filepath='%s.hd5' % learn_args['model_prefix'],
                  monitor='val_acc',
                  save_best_only=True,
                  mode='max',
                  verbose=0
              )],
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def iter_data(input_file, **learn_args):
    replay_names = [line.rstrip('\r\n') for line in open(input_file)]
    random.shuffle(replay_names)
    batch_size = learn_args['batch_size']
    for index, replay_name in enumerate(replay_names):
        try:
            replay = from_local(replay_name) if learn_args['local_replays'] \
                else from_s3(replay_name)
        except:
            continue

        if replay.num_frames < 10:
            print('Skipping:', replay_name, '#frames:', replay.num_frames)
            continue

        print('Replay:', replay_name,
              '(', (index + 1), '/', len(replay_names), ')',
              'Winner:', replay.player_names[replay.winner - 1])
        X_train, Y_train, X_test, Y_test = \
            get_train_test_data(replay, **learn_args)

        for start in range(0, X_train.shape[0], batch_size):
            begin, end = start, start + batch_size
            yield X_train[begin:end], Y_train[begin:end], True
        yield X_test, Y_test, False


def learn_from_multiple_replays(input_file, **learn_args):
    model = None
    num_samples_seen = 0
    checkpoint_samples = learn_args['checkpoint_samples']
    checkpoint_index = 0
    for epoch in range(10):
        print('>>> Epoch:', (epoch + 1))
        for X, Y, is_training in iter_data(input_file, **learn_args):
            if model is None:
                model = create_model(X, **learn_args)
            if is_training:
                model.train_on_batch(X, Y)
                num_samples_seen += X.shape[0]
                curr_checkpoint = int(num_samples_seen / checkpoint_samples)
                if curr_checkpoint > checkpoint_index:
                    checkpoint_index = curr_checkpoint
                    print('Processed:', num_samples_seen, 'samples')
                    save_model(model, **learn_args)
            else:
                score = model.evaluate(X, Y, verbose=0)
                print('Test score:', score[0])
                print('Test accuracy:', score[1])


@plac.annotations(
    input_file='Input file name',
    model_prefix='Model prefix',
    model_type=('Model type',
                'option', 'm', str, ['linear', 'mlp', 'cnn1d', 'cnn2d']),
    feature_type=('Feature type (axes, tile)',
                  'option', 'f', str, ['axes', 'tile']),
    local_replays=('Whether the hlt files are local or S3',
                   'flag', 'l'),
    learn_single=('Learn from a single replay or batch of replays',
                  'flag', 's'),
    window_size=('Window size for features',
                 'option', 'w'),
    test_size=('Proportion of the data to use for validation',
               'option'),
    batch_size=('Size of batch to use during training',
                'option'),
    checkpoint_samples=('Number of samples to checkpoint',
                        'option'),
)
def learn(
    input_file,
    model_prefix,
    model_type='mlp',
    feature_type='tile',
    local_replays=False,
    learn_single=False,
    window_size=5,
    test_size=0.1,
    batch_size=32,
    checkpoint_samples=100000,
):
    learn_args = {
        'linear': model_type in ('linear', 'mlp'),
        'model_type': model_type,
        'feature_type': feature_type,
        'model_prefix': model_prefix,
        'local_replays': local_replays,
        'window': int(window_size),
        'test_size': float(test_size),
        'batch_size': int(batch_size),
        'checkpoint_samples': int(checkpoint_samples),
    }
    print('Learning args: {}'.format(learn_args))
    with open('%s.json' % model_prefix, 'w') as fout:
        json.dump(learn_args, fout)

    if learn_single:
        learn_from_single_replay(input_file, **learn_args)
    else:
        learn_from_multiple_replays(input_file, **learn_args)


if __name__ == '__main__':
    plac.call(learn)
