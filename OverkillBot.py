from utils import hlt
from utils.overkill import get_move

my_id, game_map = hlt.get_init()
hlt.send_init("OverkillBot")

while True:
    game_map.get_frame()
    moves = [get_move(game_map, square, my_id)
             for square in game_map if square.owner == my_id]
    hlt.send_frame(moves)
