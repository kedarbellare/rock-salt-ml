import sys
stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')
from keras.utils import np_utils
sys.stdout = stdout
sys.stderr = stderr
import numpy as np

from utils.hlt import DIRECTIONS

nb_classes = len(DIRECTIONS)


def get_XY(replay, player, window, linear=False):
    X, y, w = ReplayFeaturizer(player, window).examples_and_labels(replay)
    for c in range(nb_classes):
        print("class:", c, "prior:", float(np.sum(y == c)) / len(y))
    Y = np_utils.to_categorical(y, nb_classes)
    if linear:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    return X, Y, w


def east_window(X, x, y, window):
    return X\
        .take(range(x + 1, x + window + 1), axis=1, mode='wrap')\
        .take(range(y, y + 1), axis=0, mode='wrap')\
        .reshape(window)


def west_window(X, x, y, window):
    return X\
        .take(range(x - window, x), axis=1, mode='wrap')\
        .take(range(y, y + 1), axis=0, mode='wrap')\
        .reshape(window)


def east_west_window(X, x, y, window):
    return X\
        .take(range(x - window, x + window + 1), axis=1, mode='wrap')\
        .take(range(y, y + 1), axis=0, mode='wrap')\
        .reshape(2 * window + 1)


def north_window(X, x, y, window):
    return X\
        .take(range(y - window, y), axis=0, mode='wrap')\
        .take(range(x, x + 1), axis=1, mode='wrap')\
        .reshape(window)


def south_window(X, x, y, window):
    return X\
        .take(range(y + 1, y + window + 1), axis=0, mode='wrap')\
        .take(range(x, x + 1), axis=1, mode='wrap')\
        .reshape(window)


def north_south_window(X, x, y, window):
    return X\
        .take(range(y - window, y + window + 1), axis=0, mode='wrap')\
        .take(range(x, x + 1), axis=1, mode='wrap')\
        .reshape(2 * window + 1)


def surrounding_window(X, x, y, window):
    return X\
        .take(range(y - window, y + window + 1), axis=0, mode='wrap')\
        .take(range(x - window, x + window + 1), axis=1, mode='wrap')


class ReplayFeaturizer(object):

    def __init__(self, player, window):
        self.player = player
        self.window = window

    def positions_and_examples_from_frame(self, frame):
        player_y, player_x = frame.player_yx(self.player)
        # player_positions = frame.player_positions(self.player)
        # unowned_positions = frame.unowned_positions
        competitor_positions = frame.competitor_positions(self.player)
        player_strengths = frame.player_strengths(self.player) / 255
        competitor_strengths = frame.competitor_strengths(self.player) / 255
        unowned_strengths = frame.unowned_strengths / 255
        productions = frame.productions / 51
        examples = []
        for x, y in zip(player_x, player_y):
            """
            examples.append([
                surrounding_window(player_positions, x, y, self.window),
                surrounding_window(unowned_positions, x, y, self.window),
                surrounding_window(competitor_positions, x, y, self.window),
                surrounding_window(player_strengths, x, y, self.window),
                surrounding_window(unowned_strengths, x, y, self.window),
                surrounding_window(competitor_strengths, x, y, self.window),
                surrounding_window(productions, x, y, self.window),
            ])
            """
            examples.append([
                north_south_window(player_strengths, x, y, self.window),
                east_west_window(player_strengths, x, y, self.window),
                north_south_window(productions, x, y, self.window),
                east_west_window(productions, x, y, self.window),
                north_south_window(competitor_positions, x, y, self.window),
                east_west_window(competitor_positions, x, y, self.window),
                north_south_window(competitor_strengths, x, y, self.window),
                east_west_window(competitor_strengths, x, y, self.window),
                north_south_window(unowned_strengths, x, y, self.window),
                east_west_window(unowned_strengths, x, y, self.window),
            ])
        weights = np.ones(len(examples)) / len(examples)
        return player_x, player_y, examples, weights

    def examples_from_frame(self, frame):
        _, _, examples, weights = self.positions_and_examples_from_frame(frame)
        return examples, weights

    def examples_and_labels(self, replay):
        examples, labels, weights = [], [], []
        for frame in range(replay.num_frames - 1):
            replay_frame = replay.get_frame(frame)
            egs, wts = self.examples_from_frame(replay_frame)
            examples.extend(egs)
            weights.extend(wts)
            for move in replay_frame.player_moves(self.player):
                labels.append(move)
        return np.array(examples), np.array(labels, dtype=int), \
            np.array(weights)
