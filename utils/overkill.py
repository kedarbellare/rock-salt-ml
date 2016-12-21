from utils.hlt import NORTH, EAST, SOUTH, WEST, STILL, Move


def __find_nearest_enemy_direction(game_map, square, my_id):
    direction = NORTH
    max_distance = min(game_map.width, game_map.height) / 2
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        current = square
        while current.owner == my_id and distance < max_distance:
            distance += 1
            current = game_map.get_target(current, d)
        if distance < max_distance:
            direction = d
            max_distance = distance
    return direction


def __heuristic(game_map, square, my_id):
    if square.owner == 0 and square.strength > 0:
        return square.production / square.strength
    else:
        # return total potential damage caused by overkill when
        # attacking this square
        return sum(
            neighbor.strength
            for neighbor in game_map.neighbors(square)
            if neighbor.owner not in (0, my_id)
        )


def get_move(game_map, square, my_id):
    target, direction = max(
        ((neighbor, direction)
         for direction, neighbor in enumerate(game_map.neighbors(square))
         if neighbor.owner != my_id),
        default=(None, None),
        key=lambda t: __heuristic(game_map, t[0], my_id)
    )
    if target is not None and target.strength < square.strength:
        return Move(square, direction)
    elif square.strength < square.production * 5:
        return Move(square, STILL)

    border = any(neighbor.owner != my_id
                 for neighbor in game_map.neighbors(square))
    if not border:
        return Move(
            square,
            __find_nearest_enemy_direction(game_map, square, my_id)
        )
    else:
        # wait until we are strong enough to attack
        return Move(square, STILL)
