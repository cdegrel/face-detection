from src.server import run
from src.config import Config

if __name__ == '__main__':
    run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
