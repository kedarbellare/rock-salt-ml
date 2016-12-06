"""
Frame/replay processing to create examples and labels
"""
import numpy as np

from utils.logging import logging

logger = logging.getLogger(__name__)


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


def process_frame_axes(frame, player, window):
    """
    Processes a frame along the north-south and east-west axes.

    :param frame: Halite game frame
    :param player: Player (1-indexed)
    :param window: Window size
    :returns: tuple of examples and labels
    """
    player_y, player_x = frame.player_yx(player)
    player_moves = frame.player_moves(player)

    player_strengths = frame.player_strengths(player) / 255
    competitor_strengths = frame.competitor_strengths(player) / 255
    unowned_strengths = frame.unowned_strengths / 255
    productions = frame.productions / 51

    examples, labels = [], []
    for x, y, move in zip(player_x, player_y, player_moves):
        features = [
            north_south_window(player_strengths, x, y, window),
            east_west_window(player_strengths, x, y, window),
            north_south_window(productions, x, y, window),
            east_west_window(productions, x, y, window),
            north_south_window(competitor_strengths, x, y, window),
            east_west_window(competitor_strengths, x, y, window),
            north_south_window(unowned_strengths, x, y, window),
            east_west_window(unowned_strengths, x, y, window),
        ]
        examples.append(features)
        labels.append(move)
    return examples, labels


def process_frame_tile(frame, player, window):
    """
    Processes a frame with a square window around every (x, y) position
    """
    player_y, player_x = frame.player_yx(player)
    player_moves = frame.player_moves(player)

    stacked = np.array([
        frame.player_strengths(player) / 255,
        frame.productions / 51,
        frame.competitor_strengths(player) / 255,
        frame.unowned_strengths / 255,
    ])

    examples, labels = [], []
    for x, y, move in zip(player_x, player_y, player_moves):
        features = stacked\
            .take(range(y - window, y + window + 1), axis=1, mode='wrap')\
            .take(range(x - window, x + window + 1), axis=2, mode='wrap')
        examples.append(features)
        labels.append(move)
    return examples, labels


def process_replay(frame_processor, replay, player, window):
    """
    Takes one of the frame processors defined and returns the examples and
    labels for the complete replay.
    """
    examples, labels = [], []
    for idx in range(replay.num_frames - 1):
        frame = replay.get_frame(idx)
        frame_examples, frame_labels = frame_processor(frame, player, window)
        examples.extend(frame_examples)
        labels.extend(frame_labels)
    return examples, labels
