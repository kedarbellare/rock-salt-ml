import boto
from boto.s3.key import Key
import gzip
import numpy as np
import ujson as json

conn = None


class HaliteReplayFrame(object):

    def __init__(self, productions, owners, strengths, moves):
        self.productions = productions
        self.owners = owners
        self.strengths = strengths
        self.moves = moves

    @staticmethod
    def from_game_map(game_map):
        return HaliteReplayFrame(
            productions=np.array(game_map.production),
            owners=np.array([square.owner for square in game_map])
            .reshape(game_map.height, game_map.width),
            strengths=np.array([square.strength for square in game_map])
            .reshape(game_map.height, game_map.width),
            moves=np.zeros((game_map.height, game_map.width), dtype=int)
        )

    @property
    def owned_positions(self):
        return self.owners != 0

    def player_positions(self, player):
        return self.owners == player

    def nonplayer_positions(self, player):
        return self.owners != player

    def competitor_positions(self, player):
        return self.owned_positions & (self.owners != player)

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

    def total_nonplayer_strength(self, player):
        return np.sum(self.nonplayer_strengths(player))

    def total_nonplayer_production(self, player):
        return np.sum(self.nonplayer_productions(player))

    def total_competitor_strength(self, player):
        return np.sum(self.competitor_strengths(player))

    def total_competitor_production(self, player):
        return np.sum(self.competitor_productions(player))

    def total_competitor_territory(self, player):
        return np.sum(self.competitor_positions(player))

    def __get_strengths(self, positions):
        strengths = np.zeros_like(positions, dtype=float)
        strengths[positions] = self.strengths[positions]
        return strengths

    def __get_productions(self, positions):
        productions = np.zeros_like(positions, dtype=float)
        productions[positions] = self.productions[positions]
        return productions

    def player_strengths(self, player):
        return self.__get_strengths(self.player_positions(player))

    def player_productions(self, player):
        return self.__get_productions(self.player_positions(player))

    def player_moves(self, player):
        return self.moves[self.player_positions(player)]

    def nonplayer_strengths(self, player):
        return self.__get_strengths(self.nonplayer_positions(player))

    def nonplayer_productions(self, player):
        return self.__get_productions(self.nonplayer_positions(player))

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

    @property
    def winner(self):
        last_frame = self.get_frame(self.num_frames - 1)
        _, winner = max(
            (last_frame.total_player_territory(player), player)
            for player in range(1, self.num_players + 1)
        )
        return winner

    def get_frame(self, frame):
        return HaliteReplayFrame(
            self.__productions_arr,
            self.__frames_arr[frame][:, :, 0],
            self.__frames_arr[frame][:, :, 1],
            self.__moves_arr[frame] if frame < self.num_frames - 1
            else np.zeros((self.height, self.width), dtype=int)
        )


def from_s3(fname):
    global conn
    if conn is None:
        conn = boto.connect_s3()
    bucket = conn.get_bucket('halitereplaybucket')
    k = Key(bucket)
    k.key = fname
    contents = k.get_contents_as_string()
    try:
        contents = gzip.decompress(contents)
    except OSError:
        pass
    return HaliteReplay(json.loads(contents))


def from_local(fname):
    return HaliteReplay(json.load(open(fname)))
