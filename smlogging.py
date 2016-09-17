# -*- coding: utf-8 -*-

INFO = 1
ERROR = 2
DEBUG = 4

#loglevel = DEBUG | INFO | ERROR
loglevel = ERROR

def log(level, *args):
    if (loglevel & level) != 0:
        print (" ".join(map(str, args)))

