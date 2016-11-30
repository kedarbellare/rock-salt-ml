import random

from utils import hlt
from utils.hlt import DIRECTIONS, Move

my_id, game_map = hlt.get_init()
hlt.send_init("RandomPythonBot")

while True:
    moves = []
    game_map.get_frame()
    moves = [Move(square, random.choice(DIRECTIONS))
             for square in game_map if square.owner == my_id]
    hlt.send_frame(moves)
