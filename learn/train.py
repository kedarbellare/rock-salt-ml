import numpy as np

from bisect import bisect_right
import plac
import random
import subprocess
import ujson as json

from learn.keras_utils import np_utils, models, ModelCheckpoint, Sequential
from learn.keras_utils import Activation, Convolution1D, Convolution2D, \
    Dense, Dropout, Flatten
from learn.keras_utils import Adam
from learn.features import \
    process_frame_axes, process_frame_tile, \
    process_replay
from sklearn.model_selection import ShuffleSplit
from utils.hlt import Move, Square, DIRECTIONS
from utils.logging import logging, log
from utils.replay import from_local, from_s3

nb_classes = len(DIRECTIONS)
FEATURE_TYPE_PROCESSOR = {
    'axes': process_frame_axes,
    'tile': process_frame_tile,
}
logger = logging.getLogger(__name__)


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


def create_base_model(X, **learn_args):
    if learn_args.get('input_model'):
        base_model = load_model(**learn_args)
        if len(base_model.layers) == 2:
            base_model.pop()
        return base_model
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
    return base_model


def create_model(X, **learn_args):
    base_model = create_base_model(X, **learn_args)
    model = Sequential([
        base_model,
        Activation('softmax'),
    ])
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def store_params(learn_args):
    log(logger.info, 'Learning args: {}'.format(learn_args))
    with open('%s.json' % learn_args['model_prefix'], 'w') as fout:
        json.dump(learn_args, fout)


def save_model(model, **learn_args):
    log(logger.info, '>>> Saving model...')
    model.save('%s.h5' % learn_args['model_prefix'])
    log(logger.info, 'Done.')


def load_model(**learn_args):
    if learn_args.get('input_model'):
        return models.load_model(learn_args['input_model'])
    else:
        return models.load_model('%s.h5' % learn_args['model_prefix'])


def best_moves(model, frame, player, **learn_args):
    player_y, player_x = frame.player_yx(player)
    frame_processor = FEATURE_TYPE_PROCESSOR[learn_args['feature_type']]
    examples, _ = frame_processor(frame, player, window=learn_args['window'])
    qlearning = learn_args.get('qlearn')
    X = np.array(examples)
    if learn_args['linear']:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    eps = learn_args.get('curr_eps', 0.0)
    moves = []
    for x, y, vec in zip(player_x, player_y, X):
        if qlearning and np.random.random() < eps:
            direction = random.choice(DIRECTIONS)
        else:
            pred = np.argmax(model.predict(np.array([vec]))[0])
            direction = DIRECTIONS[pred]
        moves.append(Move(Square(x, y, 0, 0, 0), direction))
    return moves


def get_XY(replay, player, **learn_args):
    # construct input and output
    examples, labels = process_replay(
        FEATURE_TYPE_PROCESSOR[learn_args['feature_type']],
        replay,
        player=player,
        window=learn_args['window']
    )
    X, y = np.array(examples), np.array(labels, dtype=int)
    Y = np_utils.to_categorical(y, nb_classes=nb_classes)
    if learn_args['linear']:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    return X, Y


def get_train_test_data(replay, player, **learn_args):
    X, Y = get_XY(replay, player, **learn_args)

    # create splits for train/test
    kf = ShuffleSplit(
        n_splits=1,
        test_size=learn_args['test_size'],
        random_state=0
    )
    train_index, test_index = next(kf.split(X))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    log(logger.info, '#train: {}'.format(X_train.shape))
    log(logger.info, '#test: {}'.format(X_test.shape))
    return X_train, Y_train, X_test, Y_test


def learn_from_single_replay(input_file, **learn_args):
    replay = from_local(input_file) if learn_args['local_replays'] else \
        from_s3(input_file)

    X_train, Y_train, X_test, Y_test = \
        get_train_test_data(replay, replay.winner, **learn_args)

    # create model
    model = create_model(X_train, **learn_args)
    model.fit(X_train, Y_train, nb_epoch=20, verbose=1,
              callbacks=[ModelCheckpoint(
                  filepath='%s.h5' % learn_args['model_prefix'],
                  monitor='val_acc',
                  save_best_only=True,
                  mode='max',
                  verbose=0
              )],
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    log(logger.info, 'Test score: {} accuracy: {}'.format(score[0], score[1]))


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
            log(logger.info, 'Skipping: {}, #frames={}'.format(
                replay_name, replay.num_frames))
            continue

        log(logger.info, 'Replay: {} ({} / {}) Winner: {}'.format(
            replay_name, (index + 1), len(replay_names),
            replay.player_names[replay.winner - 1]))
        X_train, Y_train, X_test, Y_test = \
            get_train_test_data(replay, replay.winner, **learn_args)

        for start in range(0, X_train.shape[0], batch_size):
            begin, end = start, start + batch_size
            yield X_train[begin:end], Y_train[begin:end], True
        yield X_test, Y_test, False


def learn_from_multiple_replays(input_file, **learn_args):
    model = None
    num_samples_seen = 0
    checkpoint_samples = learn_args['checkpoint_samples']
    checkpoint_index = 0
    for epoch in range(learn_args['epochs']):
        log(logger.info, '>>> Epoch: {}'.format(epoch + 1))
        for X, Y, is_training in iter_data(input_file, **learn_args):
            if model is None:
                model = create_model(X, **learn_args)
            if is_training:
                model.train_on_batch(X, Y)
                num_samples_seen += X.shape[0]
                curr_checkpoint = int(num_samples_seen / checkpoint_samples)
                if curr_checkpoint > checkpoint_index:
                    checkpoint_index = curr_checkpoint
                    log(logger.info, 'Processed: {} samples'.format(
                        num_samples_seen))
                    save_model(model, **learn_args)
            else:
                score = model.evaluate(X, Y, verbose=0)
                log(logger.info, 'Test score: {} accuracy: {}'.format(
                    score[0], score[1]))


def __qlearn_model(model, X, Y, territories, rewards, **learn_args):
    assert sum(territories) == X.shape[0]
    cumulative_territories = [
        sum(territories[:i]) for i in range(1, len(territories) + 1)
    ]
    discount = learn_args['gamma']
    indices = list(range(X.shape[0]))
    random.shuffle(indices)
    batch_size = learn_args['batch_size']
    loss = 0.0
    for start in range(0, X.shape[0], batch_size):
        begin, end = start, start + batch_size
        curr_indices = indices[begin:end]
        X_batch = X[curr_indices]
        Y_batch = Y[curr_indices]
        Q_scores = model.predict(X)
        batch_rewards = np.zeros_like(Y_batch, dtype=float)
        batch_rewards += Q_scores[curr_indices]
        for i, idx in enumerate(curr_indices):
            frame = bisect_right(cumulative_territories, idx)
            frame_begin = 0 if frame == 0 else \
                cumulative_territories[frame - 1]
            frame_end = cumulative_territories[frame]
            step_reward = rewards[frame]
            if frame < len(territories) - 1:
                next_frame_begin = frame_end
                next_frame_end = cumulative_territories[frame + 1]
                next_frame_q = Q_scores[next_frame_begin:next_frame_end]
                step_reward += discount * np.sum(np.max(next_frame_q, axis=1))
            frame_q = Q_scores[frame_begin:frame_end] * \
                Y[frame_begin:frame_end]
            step_reward -= np.sum(frame_q)
            step_reward /= territories[frame]
            batch_rewards[i] += Y_batch[i] * step_reward
        loss += model.train_on_batch(X_batch, batch_rewards)
    return loss


def learn_from_qlearning(**learn_args):
    start_eps = learn_args['start_eps']
    end_eps = learn_args['end_eps']
    nb_epochs = learn_args['epochs']
    eps_delta = (start_eps - end_eps) / nb_epochs
    wins = 0
    observe = 0
    curr_eps = start_eps
    model = None
    player = 1  # the first player is always us
    for epoch in range(nb_epochs):
        learn_args['curr_epoch'] = epoch
        learn_args['curr_eps'] = curr_eps
        store_params(learn_args)

        # play the game
        proc = subprocess.run(
            './halite -d "30 30" -t -q '
            '"python3 MyBot.py" "python3 OverkillBot.py"',
            shell=True,
            check=True,
            stdout=subprocess.PIPE
        )
        game_outputs = proc.stdout.decode('utf-8').split('\n')
        log(logger.info, '\n'.join(game_outputs))
        replay_name, _ = game_outputs[2].split()
        if game_outputs[3] == '1 1':
            wins += 1

        # get the replay
        replay = from_local(replay_name)
        if replay.num_frames < 10:
            log(logger.info, 'Skipping: {}, #frames={}'.format(
                replay_name, replay.num_frames))
            continue

        X, Y = get_XY(replay, player, **learn_args)
        if model is None:
            model = create_base_model(X, **learn_args)
            optimizer = Adam(lr=1e-8)
            model.compile(loss='mse', optimizer=optimizer)

        rewards = []
        territories = []
        map_size = 1.  # * replay.width * replay.height
        for i in range(replay.num_frames - 1):
            frame = replay.get_frame(i)
            next_frame = replay.get_frame(i + 1)
            territories.append(int(frame.total_player_territory(player)))
            frame_reward = \
                frame.total_player_strength(player) / 255 + \
                frame.total_player_production(player) / 20
            next_frame_reward = \
                next_frame.total_player_strength(player) / 255 + \
                next_frame.total_player_production(player) / 20
            rewards.append(
                (1. * (next_frame_reward - frame_reward)) / map_size)
        # additional reward signal if player won/lost (is this needed???)
        rewards[-1] += 1.0 if player == replay.winner else -1.0
        log(logger.info, '#frames={}, max territory={}'.format(
            replay.num_frames, max(territories)))
        log(logger.info, rewards)

        loss = __qlearn_model(model, X, Y, territories, rewards, **learn_args)
        save_model(model, **learn_args)
        log(logger.info, "Epoch {}/{} | Loss {:.4f} | Win count {}".format(
            epoch + 1, nb_epochs, loss, wins))

        if curr_eps > end_eps and epoch >= observe:
            curr_eps -= eps_delta


@plac.annotations(
    input_file='Input file name',
    model_prefix='Model prefix',
    input_model=('Input model file name (if any)', 'option', 'i'),
    model_type=('Model type', 'option', 'm', str,
                ['linear', 'mlp', 'cnn1d', 'cnn2d']),
    feature_type=('Feature type (axes, tile)', 'option', 'f', str,
                  ['axes', 'tile']),
    local_replays=('Whether the hlt files are local or S3', 'flag', 'l'),
    learn_single=('Learn from a single or a batch of replays', 'flag', 's'),
    qlearn=('Learn via q-learning', 'flag', 'q'),
    window_size=('Window size for features', 'option', 'w'),
    test_size=('Proportion of the data to use for validation', 'option'),
    batch_size=('Size of batch to use during training', 'option'),
    checkpoint_samples=('Number of samples to checkpoint', 'option'),
    gamma=('Discounting factor for future rewards', 'option'),
    start_eps=('Starting epsilon', 'option'),
    end_eps=('Ending epsilon', 'option'),
    epochs=('Number of training epochs', 'option'),
)
def learn(
    input_file,
    model_prefix,
    input_model=None,
    model_type='mlp',
    feature_type='tile',
    local_replays=False,
    learn_single=False,
    qlearn=False,
    window_size=5,
    test_size=0.1,
    batch_size=32,
    checkpoint_samples=100000,
    gamma=0.9,
    start_eps=0.5,
    end_eps=0.1,
    epochs=1000,
):
    learn_args = {
        'linear': model_type in ('linear', 'mlp'),
        'model_type': model_type,
        'input_model': input_model,
        'feature_type': feature_type,
        'model_prefix': model_prefix,
        'local_replays': local_replays,
        'qlearn': qlearn,
        'window': int(window_size),
        'test_size': float(test_size),
        'batch_size': int(batch_size),
        'checkpoint_samples': int(checkpoint_samples),
        'gamma': float(gamma),
        'start_eps': float(start_eps),
        'end_eps': float(end_eps),
        'epochs': int(epochs),
    }
    store_params(learn_args)

    if qlearn:
        learn_from_qlearning(**learn_args)
    else:
        if learn_single:
            learn_from_single_replay(input_file, **learn_args)
        else:
            learn_from_multiple_replays(input_file, **learn_args)


if __name__ == '__main__':
    plac.call(learn)
