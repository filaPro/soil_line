import os
import sys


def load_proj():
    if getattr(sys, 'frozen', False):
        os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj')
        print(os.listdir(os.path.join(sys._MEIPASS, 'proj')))
    elif sys.platform == 'win32':
        os.environ['PROJ_LIB'] = 'C:\\ProgramData\\Miniconda3\\envs\\soil_line\\Library\\share\\proj'  # ?

