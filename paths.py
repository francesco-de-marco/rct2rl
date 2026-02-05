import platform
import os

if 'macOS' in platform.platform():
    
    RCT_EXECUTABLE = "/Users/jcampbell/Projects/Apps/openrct2-pathrl/build/OpenRCT2.app/Contents/MacOS/OpenRCT2"
    BASE_DIR = "/Users/jcampbell/Projects/Research/park_gen/pathrl/"
else:
    
    RCT_EXECUTABLE = "/home/francesco/ML/OpenRCT2-fork/build/openrct2"
    
    
    BASE_DIR = "/home/francesco/ML/rctrl/"

PARK_PATH = BASE_DIR

PARK_DIR = f"{BASE_DIR}small_parks"
