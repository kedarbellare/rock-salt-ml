import numpy as np

from bisect import bisect_right
import plac
import random
import subprocess
import ujson as json

from learn.keras_utils import np_utils, models, Sequential
from learn.keras_utils import model_from_yaml
from learn.keras_utils import Activation, Convolution1D, Convolution2D, \
    Dense, Dropout, Flatten
from learn.keras_utils import Adam
from learn.features import \
    process_frame_axes, process_frame_tile, process_frame_tile_symmetric, \
    process_replay
from sklearn.model_selection import ShuffleSplit
from utils.hlt import Move, Square, DIRECTIONS, STILL
from utils.halite_logging import logging, log
from utils.overkill import get_move
from utils.replay import from_local, from_s3, to_s3

nb_classes = len(DIRECTIONS)
FEATURE_TYPE_PROCESSOR = {
    'axes': process_frame_axes,
    'tile': process_frame_tile,
    'sym_tile': process_frame_tile_symmetric,
}
logger = logging.getLogger(__name__)
np.set_printoptions(precision=3, linewidth=120)


def get_linear_model(input_shape):
    return Sequential([
        Dense(nb_classes, input_shape=input_shape),
    ])


def get_mlp_model(input_shape):
    return Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dropout(0.1),
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
        Convolution2D(128, 2, 2,
                      border_mode='valid', dim_ordering='th',
                      activation='relu', input_shape=input_shape),
        Dropout(0.1),
        Convolution2D(64, 2, 2,
                      border_mode='valid', dim_ordering='th',
                      activation='relu'),
        Dropout(0.1),
        Convolution2D(32, 2, 2,
                      border_mode='valid', dim_ordering='th',
                      activation='relu'),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
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
    scale = learn_args.get('y_scale')
    model = create_base_model(X, **learn_args)
    if scale > 0.0:
        model.compile(loss='mse', optimizer=Adam(1e-5))
    else:
        model = Sequential([model, Activation('softmax')])
        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(1e-3),
            metrics=['accuracy']
        )
    return model


def store_params(learn_args):
    log(logger.info, 'Learning args: {}'.format(learn_args))
    with open('%s.json' % learn_args['model_prefix'], 'w') as fout:
        json.dump(learn_args, fout)


def save_model(model, **learn_args):
    log(logger.info, '>>> Saving model...')
    with open('%s.yaml' % learn_args['model_prefix'], 'w') as fout:
        fout.write(model.to_yaml())
    model.save_weights('%s.h5' % learn_args['model_prefix'])
    log(logger.info, 'Done.')


def load_model(**learn_args):
    if learn_args.get('input_model'):
        with open('%s.yaml' % learn_args['model_prefix']) as fin:
            model = model_from_yaml(fin.read())
        model.load_weights(learn_args['input_model'])
        return model
    else:
        with open('%s.yaml' % learn_args['model_prefix']) as fin:
            model = model_from_yaml(fin.read())
        model.load_weights('%s.h5' % learn_args['model_prefix'])
        return model


def best_moves(model, game_map, frame, player, **learn_args):
    player_y, player_x = frame.player_yx(player)
    player_borders = frame.borders(player)
    frame_processor = FEATURE_TYPE_PROCESSOR[learn_args['feature_type']]
    examples, _ = frame_processor(frame, player, window=learn_args['window'])
    qlearning = learn_args.get('qlearn')
    X = np.array(examples)
    if learn_args['linear']:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    eps = learn_args.get('curr_eps', 0.0)
    best_indices = model.predict(X).argmax(axis=1)
    if learn_args.get('rnd_still_moves'):
        still_prob = .6
        dir_probs = [
            still_prob if d == STILL else (1 - still_prob) / 4
            for d in DIRECTIONS
        ]
    else:
        dir_probs = None

    curr_epoch = learn_args.get('curr_epoch', 0)
    xy_to_square = {
        (sq.x, sq.y): sq for sq in game_map if sq.owner == player
    }
    moves = []
    for x, y, i in zip(player_x, player_y, best_indices):
        direction = DIRECTIONS[i]
        square = xy_to_square[(x, y)]
        if qlearning:
            is_border = player_borders[y, x]
            xy_eps = eps * 1.2 if is_border else eps
            if np.random.random() < xy_eps:
                if curr_epoch < learn_args.get('observe'):
                    overkill_move = get_move(game_map, square, player)
                    direction = overkill_move.direction
                else:
                    direction = np.random.choice(DIRECTIONS, p=dir_probs)
        moves.append(Move(square, direction))
    return moves


def get_XY(replay, player, **learn_args):
    # construct input and output
    examples, labels = process_replay(
        FEATURE_TYPE_PROCESSOR[learn_args['train_feature_type']],
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
    scale = learn_args.get('y_scale')
    X, Y = get_XY(replay, player, **learn_args)
    if scale is not None and scale > 0.0:
        Y = scale * (2 * Y - 1)

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
    model.fit(X_train, Y_train, nb_epoch=10, verbose=1,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    log(logger.info, 'Test score: {}'.format(score))
    save_model(model, **learn_args)


def iter_data(input_file, **learn_args):
    replay_names = [line.rstrip('\r\n') for line in open(input_file)]
    random.shuffle(replay_names)
    batch_size = learn_args['batch_size']
    for index, replay_name in enumerate(replay_names):
        replay = from_local(replay_name) if learn_args['local_replays'] \
                else from_s3(replay_name)

        if replay.num_frames < 10:
            log(logger.info, 'Skipping: {}, #frames={}'.format(
                replay_name, replay.num_frames))
            continue

        winner = replay.winner
        player = winner
        player_name = learn_args.get('player_name')
        player_names = replay.player_names
        if player_name and player_name not in player_names:
            continue
        elif player_name and player_name in player_names:
            player = 1 + player_names.index(player_name)

        log(logger.info, 'Replay: {} ({}/{}) Players: {} '
            'Winner: {} Player: {}'.format(
                replay_name, (index + 1), len(replay_names),
                player_names,
                player_names[winner - 1],
                player_name,
        ))
        X_train, Y_train, X_test, Y_test = \
            get_train_test_data(replay, player, **learn_args)

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
                    # save model and replay log to s3
                    to_s3('models', '%s.h5' % learn_args['model_prefix'])
                    to_s3('replays', 'replay.log')
            else:
                score = model.evaluate(X, Y, verbose=0)
                log(logger.info, 'Test score: {}'.format(score))


def __qlearn_model(model, X, Y, territories, rewards, **learn_args):
    import sys

    assert sum(territories) == X.shape[0]
    cumulative_territories = [
        sum(territories[:i]) for i in range(1, len(territories) + 1)
    ]
    frame_ends = cumulative_territories
    frame_begins = [0] + cumulative_territories[:-1]
    discount = learn_args['gamma']
    num_frames = len(territories)
    batch_size = learn_args['batch_size']
    indices = list(range(X.shape[0]))
    loss = 0.0
    num_samples = 0
    max_memory = learn_args['max_memory']

    for epoch in range(1):
        Q_scores = model.predict(X)
        max_Q_scores = Q_scores.max(axis=1)
        move_Q_scores = Q_scores * Y
        random.shuffle(indices)

        for frame_i in range(num_frames):
            mem_begin = max(0, frame_i + 1 - max_memory)
            mem_end = frame_i + 1
            curr_indices = [
                np.random.randint(frame_begins[frame],
                                  frame_ends[frame])
                for frame in np.random.choice(
                    np.arange(mem_begin, mem_end),
                    replace=False,
                    size=min(batch_size, mem_end - mem_begin)
                )
            ]
            """
            for batch_start in range(0, len(indices), batch_size):
            curr_indices = indices[batch_start:batch_start + batch_size]
            """
            X_batch = X[curr_indices]
            Y_batch = Y[curr_indices]
            batch_rewards = 1. * Q_scores[curr_indices]
            num_samples += len(curr_indices)
            for i, idx in enumerate(curr_indices):
                frame = bisect_right(cumulative_territories, idx)
                frame_begin = frame_begins[frame]
                frame_end = frame_ends[frame]
                step_reward = rewards[frame]
                if frame < num_frames - 1:
                    next_frame_begin = frame_begins[frame + 1]
                    next_frame_end = frame_ends[frame + 1]
                    next_frame_q_max = \
                        max_Q_scores[next_frame_begin:next_frame_end]
                    step_reward += discount * next_frame_q_max.sum()
                frame_q = move_Q_scores[frame_begin:frame_end]
                step_reward -= frame_q.sum()
                batch_rewards[i] += Y_batch[i] * step_reward
            loss += model.train_on_batch(X_batch, batch_rewards)
            if num_samples % 50 == 0:
                print('Loss: {:.4f} #samples={} ...  '.format(
                    loss, num_samples), end='')
                sys.stdout.flush()
    print('Loss: {:.4f} #samples={} ...  '.format(
        loss, num_samples))
    return loss


def learn_from_qlearning(**learn_args):
    start_eps = learn_args['start_eps']
    end_eps = learn_args['end_eps']
    nb_epochs = learn_args['epochs']
    observe = nb_epochs / 10
    eps_delta = (start_eps - end_eps) / observe
    curr_eps = start_eps
    model = None
    player = 1  # the first player is always us
    wins = 0  # count the number of times player wins
    for epoch in range(nb_epochs):
        learn_args['curr_epoch'] = epoch
        learn_args['curr_eps'] = curr_eps
        store_params(learn_args)

        # play the game
        # TODO: revert use of fixed seed
        proc = subprocess.run(
            './halite -d "30 30" -t -q -s 1559383297 '
            '"THEANO_FLAGS=device=cpu,floatX=float32 python3 MyBot.py" '
            '"python3 OverkillBot.py"',
            shell=True,
            check=True,
            stdout=subprocess.PIPE
        )
        game_outputs = proc.stdout.decode('utf-8').split('\n')
        log(logger.info, '\n'.join(game_outputs))
        replay_name, _ = game_outputs[2].split()
        if game_outputs[3].startswith('1 1'):
            wins += 1
            # save hlt to replays
            # to_s3('replays', replay_name)

        # get the replay
        replay = from_local(replay_name)
        if replay.num_frames < 10:
            log(logger.info, 'Skipping: {}, #frames={}'.format(
                replay_name, replay.num_frames))
            continue

        X, Y = get_XY(replay, player, **learn_args)
        if model is None:
            model = create_base_model(X, **learn_args)
            model.compile(loss='mse', optimizer=Adam(lr=1e-9))

        rewards = []
        territories = []
        map_size = 3. * replay.width * replay.height
        for i in range(replay.num_frames - 1):
            frame = replay.get_frame(i)
            next_frame = replay.get_frame(i + 1)
            territories.append(int(frame.total_player_territory(player)))
            frame_production = \
                frame.total_player_production(player) / 20
            next_frame_production = \
                next_frame.total_player_production(player) / 20
            frame_strength = \
                frame.total_player_strength(player) / 255
            next_frame_strength = \
                next_frame.total_player_strength(player) / 255
            frame_territory = \
                frame.total_player_territory(player)
            next_frame_territory = \
                next_frame.total_player_territory(player)
            rewards.append(
                (next_frame_territory - frame_territory) +
                (next_frame_strength - frame_strength) +
                (next_frame_production - frame_production)
            )
        rewards = np.array(rewards, dtype=np.float)
        rewards /= map_size
        log(logger.info, '#frames={}, max territory={}, input={}'.format(
            replay.num_frames, max(territories), X.shape))
        log(logger.info, rewards)

        loss = __qlearn_model(model, X, Y, territories, rewards, **learn_args)
        save_model(model, **learn_args)
        log(logger.info, "Epoch {}/{} | Loss {:.4f} | Win count {}".format(
            epoch + 1, nb_epochs, loss, wins))

        if (epoch + 1) % 200 == 0:
            # save model and replay log to s3
            to_s3('models', '%s.h5' % learn_args['model_prefix'])
            # to_s3('replays', 'replay.log')

        if curr_eps > end_eps and epoch < observe:
            curr_eps -= eps_delta

        subprocess.run('rm *.hlt *.hlt.gz', shell=True)


@plac.annotations(
    input_file='Input file name',
    model_prefix='Model prefix',
    input_model=('Input model file name (if any)', 'option', 'i'),
    model_type=('Model type', 'option', 'm', str,
                ['linear', 'mlp', 'cnn1d', 'cnn2d']),
    feature_type=('Feature type (axes, tile)', 'option', 'f', str,
                  ['axes', 'tile']),
    train_feature_type=('Training feature type (axes, tile, sym_tile)',
                        'option', 'tf', str, ['axes', 'tile', 'sym_tile']),
    player_name=('Player name whose replays to train on', 'option', 'p'),
    local_replays=('Whether the hlt files are local or S3', 'flag', 'l'),
    learn_single=('Learn from a single or a batch of replays', 'flag', 's'),
    qlearn=('Learn via q-learning', 'flag', 'q'),
    window_size=('Window size for features', 'option', 'w'),
    test_size=('Proportion of the data to use for validation', 'option'),
    batch_size=('Size of batch to use during training', 'option'),
    checkpoint_samples=('Number of samples to checkpoint', 'option'),
    observe=('Number of samples to observe', 'option'),
    gamma=('Discounting factor for future rewards', 'option'),
    start_eps=('Starting epsilon', 'option'),
    end_eps=('Ending epsilon', 'option'),
    epochs=('Number of training epochs', 'option'),
    max_memory=('Size of replay memory', 'option'),
    y_scale=('Scaling of move output for learning', 'option'),
    rnd_still_moves=('Bias random moves towards STILL (instead of uniform)',
                     'flag', 'u'),
)
def learn(
    input_file,
    model_prefix,
    input_model=None,
    player_name=None,
    model_type='mlp',
    feature_type='tile',
    train_feature_type='tile',
    local_replays=False,
    learn_single=False,
    qlearn=False,
    window_size=5,
    test_size=0.1,
    batch_size=64,
    checkpoint_samples=100000,
    gamma=0.9,
    start_eps=0.5,
    end_eps=0.1,
    epochs=1000,
    observe=5000,
    max_memory=20,
    y_scale=0.0,
    rnd_still_moves=False,
):
    learn_args = {
        'linear': model_type in ('linear', 'mlp'),
        'model_type': model_type,
        'input_model': input_model,
        'feature_type': feature_type,
        'train_feature_type': train_feature_type,
        'player_name': player_name,
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
        'observe': int(observe),
        'max_memory': int(max_memory),
        'y_scale': float(y_scale),
        'rnd_still_moves': rnd_still_moves,
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
