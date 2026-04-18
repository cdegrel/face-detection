from main import __name__ as main_name, run
from config import Config

if __name__ == '__main__':
    run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
