import logging
import ujson as json

from learn.classify import load_model, best_moves
from utils.hlt import get_init, send_init, send_frame
from utils.replay import HaliteReplayFrame

logger = logging.getLogger(__name__)

logger.info('Loading model...')
learn_args = json.load(open('model.json'))
model = load_model(**learn_args)
logger.info('Done.')
logger.info('Initializing...')
my_id, game_map = get_init()
logger.info('Done.')

frame = HaliteReplayFrame.from_game_map(game_map)
best_moves(model, frame, my_id, **learn_args)

send_init("rocksalt v1")

while True:
    moves = []
    game_map.get_frame()
    frame = HaliteReplayFrame.from_game_map(game_map)
    send_frame(best_moves(model, frame, my_id, **learn_args))
