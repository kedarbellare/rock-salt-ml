import boto
from boto.s3.key import Key
import gzip
import numpy as np
import ujson as json

from utils.hlt import Location, Move, Site, GameMap, STILL

conn = boto.connect_s3()


class HaliteReplayFrame(object):

    def __init__(self, productions, sites, moves):
        self.productions = productions
        self.sites = sites
        self.moves = moves

    @property
    def owned_positions(self):
        return self.sites[:, :, 0] != 0

    def player_positions(self, player):
        return self.sites[:, :, 0] == player

    def competitor_positions(self, player):
        return self.owned_positions & (self.sites[:, :, 0] != player)

    @property
    def unowned_positions(self):
        return self.player_positions(0)

    def player_yx(self, player):
        return np.where(self.player_positions(player))

    def total_player_territory(self, player):
        return np.sum(self.player_positions(player))

    def total_player_strength(self, player):
        return np.sum(self.player_strengths(player))

    def total_player_production(self, player):
        return np.sum(self.player_productions(player))

    def __get_strengths(self, positions):
        strengths = np.zeros(positions.shape)
        strengths[positions] = self.sites[positions][:, 1]
        return strengths

    def __get_productions(self, positions):
        productions = np.zeros(positions.shape)
        productions[positions] = self.productions[positions]
        return productions

    def player_strengths(self, player):
        return self.__get_strengths(self.player_positions(player))

    def player_productions(self, player):
        return self.__get_productions(self.player_positions(player))

    def player_moves(self, player):
        if self.moves is None:
            return None
        return self.moves[self.player_positions(player)]

    @property
    def unowned_strengths(self):
        return self.__get_strengths(self.unowned_positions)

    @property
    def unowned_productions(self):
        return self.__get_productions(self.unowned_positions)

    def competitor_strengths(self, player):
        return self.__get_strengths(self.competitor_positions(player))

    def competitor_productions(self, player):
        return self.__get_productions(self.competitor_positions(player))


class HaliteReplay(object):

    def __init__(self, data):
        self.data = data
        self.__productions_arr = np.array(self.productions)
        self.__frames_arr = np.array(self.frames)
        self.__moves_arr = np.array(self.moves)

    def __getattr__(self, name):
        return self.data[name]

    def get_game_map(self, frame):
        assert 0 <= frame < self.num_frames
        game_map = GameMap(self.width, self.height, self.num_players)

        # overwrite the contents
        game_map.contents = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                owner, strength = self.frames[frame][y][x]
                row.append(Site(
                    owner=owner,
                    strength=strength,
                    production=self.productions[y][x]
                ))
            game_map.contents.append(row)
        return game_map

    def get_moves(self, frame):
        assert 0 <= frame < self.num_frames

        moves = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(Move(
                    Location(x, y),
                    direction=self.moves[frame - 1][y][x] if frame else STILL
                ))
            moves.append(row)
        return moves

    def get_frame(self, frame):
        return HaliteReplayFrame(
            self.__productions_arr,
            self.__frames_arr[frame],
            self.__moves_arr[frame] if frame < self.num_frames - 1 else None
        )


def from_s3(fname):
    bucket = conn.get_bucket('halitereplaybucket')
    k = Key(bucket)
    k.key = fname
    return HaliteReplay(
        json.loads(gzip.decompress(k.get_contents_as_string()))
    )


def from_local(fname):
    return HaliteReplay(json.load(open(fname)))


def matrix_window(X, x, y, window):
    return X.take(range(x - window, x + window + 1), axis=1, mode='wrap')\
        .take(range(y - window, y + window + 1), axis=0, mode='wrap')
