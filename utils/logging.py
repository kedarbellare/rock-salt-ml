import logging

formatstr = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
logging.basicConfig(
    filename='replay.log',
    format=formatstr,
    level=logging.INFO
)


def log(logfn, *args):
    logfn(*args)
    print(*args)
