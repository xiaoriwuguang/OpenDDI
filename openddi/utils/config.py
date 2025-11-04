import os

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..', '..', '..'))
LOG_PATH = CODE_DIR + '/data'