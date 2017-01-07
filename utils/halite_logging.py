import logging

formatstr = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
logging.basicConfig(
    filename='replay.log',
    format=formatstr,
    level=logging.INFO
)


def log(logfn, msg):
    logfn(msg)
    print(msg)
