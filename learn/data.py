from keras.utils import np_utils
import numpy as np

from utils.replay import matrix_window


def get_XY(replay, window, linear=False):
    nb_classes = 5
    X, y = ReplayData(replay, window).examples_and_labels()
    X /= 255
    Y = np_utils.to_categorical(y, nb_classes)
    if linear:
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    return X, Y


class ReplayData(object):

    def __init__(self, replay, window):
        self.replay = replay
        self.window = window

        # find the player that won
        last_frame = replay.get_frame(replay.num_frames - 1)
        _, self.player = max(
            (last_frame.total_player_territory(player), player)
            for player in range(1, replay.num_players + 1)
        )
        print('Winner({}): {}'.format(
            self.player, replay.player_names[self.player - 1]
        ))

    def examples_and_labels(self):
        examples, labels = [], []
        for frame in range(self.replay.num_frames - 1):
            replay_frame = self.replay.get_frame(frame)
            player_y, player_x = replay_frame.player_yx(self.player)
            player_moves = replay_frame.player_moves(self.player)
            player_strengths = replay_frame.player_strengths(self.player)
            player_productions = replay_frame.player_productions(self.player)
            unowned_strengths = replay_frame.unowned_strengths
            unowned_productions = replay_frame.unowned_productions
            competitor_strengths = replay_frame.competitor_strengths(
                self.player)
            competitor_productions = replay_frame.competitor_productions(
                self.player)
            for x, y, move in zip(player_x, player_y, player_moves):
                features = [
                    matrix_window(player_strengths, x, y, self.window),
                    matrix_window(player_productions, x, y, self.window),
                    matrix_window(unowned_strengths, x, y, self.window),
                    matrix_window(unowned_productions, x, y, self.window),
                    matrix_window(competitor_strengths, x, y, self.window),
                    matrix_window(competitor_productions, x, y, self.window)
                ]
                examples.append(features)
                labels.append(move)
        return np.array(examples), np.array(labels, dtype=int)
