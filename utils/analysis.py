from collections import Counter
from utils.replay import from_s3
import sys


if __name__ == '__main__':
    fnames = open(sys.argv[1]).read().split('\n')
    cnts = Counter()
    for i, fname in enumerate(fnames):
        index = i + 1
        if index % 5 == 0:
            print(fname)
        replay = from_s3(fname)
        for player in replay.player_names:
            cnts[player] += 1
        if index % 50 == 0:
            print(index, cnts.most_common(5))
