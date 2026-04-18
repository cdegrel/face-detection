#!/usr/bin/env python3
from server import run
from config import Config

if __name__ == '__main__':
    run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
