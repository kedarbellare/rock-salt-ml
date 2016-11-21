import sys
stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')

import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda, merge
from keras.utils import np_utils
sys.stdout = stdout
sys.stderr = stderr

from utils.hlt import DIRECTIONS, STILL, Location, Move, random
from utils.replay import from_local, matrix_window


nb_classes = len(DIRECTIONS)
window = 1


def save_model(model):
    model.save_weights('model.ser')
    with open('model.json', 'w') as fout:
        fout.write(model.to_json())


def load_model():
    with open('model.json') as fin:
        model = model_from_json(fin.read())
    model.compile(loss='mse', optimizer='adam')
    model.load_weights('model.ser')
    return model


def output_loss(x):
    return K.dot(K.sum(x, axis=1), K.ones_like(x))


def get_XY(frame, player):
    player_positions = frame.player_positions(player)
    nonplayer_positions = frame.nonplayer_positions(player)
    player_strengths = frame.player_strengths(player) / 255
    nonplayer_strengths = frame.nonplayer_strengths(player) / 255
    productions = frame.productions / 255
    player_y, player_x = frame.player_yx(player)
    player_moves = frame.player_moves(player)
    examples = []
    for x, y in zip(player_x, player_y):
        examples.append([
            matrix_window(player_positions, x, y, window),
            matrix_window(player_strengths, x, y, window),
            matrix_window(nonplayer_positions, x, y, window),
            matrix_window(nonplayer_strengths, x, y, window),
            matrix_window(productions, x, y, window),
        ])
    X = np.array(examples)
    X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    Y = np_utils.to_categorical(player_moves, nb_classes)
    return X, Y


def best_moves(model, frame, player):
    player_y, player_x = frame.player_yx(player)
    X, Y = get_XY(frame, player)
    moves = []
    for i in range(X.shape[0]):
        best_output = float('-inf')
        best_move = None
        for d in DIRECTIONS:
            output = model.predict([
                np.array([X[i]]),
                np_utils.to_categorical(
                    np.array([d], dtype=int),
                    nb_classes
                )
            ])
            if output > best_output:
                best_output = output
                best_move = d
        if best_output < 0:
            best_move = STILL
        # print('Best:', best_output, 'Direction:', best_move)
        loc = Location(player_x[i], player_y[i])
        moves.append(Move(loc, best_move))
    return moves


if __name__ == '__main__':
    replay = from_local(sys.argv[1])
    map_input = Input(shape=(5 * (2 * window + 1) ** 2,), name='map_input')
    map_scores = Dense(nb_classes, name='move_scores')(map_input)
    move_input = Input(shape=(nb_classes,), name='move_input')
    dot_prod = merge([map_scores, move_input], mode='dot', dot_axes=1)
    dot_prod_sum = Lambda(output_loss)(dot_prod)
    model = Model(input=[map_input, move_input], output=[dot_prod_sum])
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    inputs = []
    outputs = []
    sample_weights = []
    for i in range(1, replay.num_frames):
        # for i in range(1, 60):
        frame = replay.get_frame(i - 1)
        next_frame = replay.get_frame(i)
        for player in range(1, replay.num_players + 1):
            # for player in range(3, 4):
            territory = frame.total_player_territory(player)
            if territory == 0:
                continue
            reward = next_frame.total_player_strength(player) + \
                next_frame.total_player_production(player) - \
                frame.total_player_strength(player) - \
                frame.total_player_production(player)
            reward /= 255
            # print('Player:', player, 'Reward:', reward)
            # print('Player:', player, 'Territory:', territory)
            X, Y = get_XY(frame, player)
            inputs.append({'map_input': X, 'move_input': Y})
            outputs.append(np.ones(territory) * reward)
            sample_weights.append(np.ones(territory) / territory)
            """
            model.fit(
                {'map_input': X, 'move_input': Y},
                np.ones(territory) * reward,
                nb_epoch=1,
                verbose=0,
                sample_weight=np.ones(territory) / territory
            )
            """
    indices = list(range(len(inputs)))
    for epoch in range(10):
        random.shuffle(indices)
        for i in indices:
            model.train_on_batch(
                inputs[i], outputs[i], sample_weight=sample_weights[i])
    best_moves(model, replay.get_frame(0), 3)
    save_model(model)
