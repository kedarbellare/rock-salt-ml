import ujson as json
from urllib import request, parse


def get_games(userID, limit=500):
    games_data = []
    while len(games_data) < limit:
        params = {
            'userID': userID,
            'limit': min(20, limit - len(games_data))
        }
        if games_data:
            params['startingID'] = games_data[-1]['gameID']
        with request.urlopen('https://halite.io/api/web/game?{}'.format(
            parse.urlencode(params)
        )) as fp:
            new_games_data = json.loads(fp.read())
            if len(new_games_data) == 0:
                break
            games_data.extend(new_games_data)
    return games_data


def get_game_replays(userID, limit=500):
    return [game['replayName'] for game in get_games(userID, limit=limit)]


if __name__ == '__main__':
    for replayName in get_game_replays(1017, limit=2000):
        print(replayName)
