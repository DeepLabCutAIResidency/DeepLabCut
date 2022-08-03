from cgi import print_arguments
import time
import random
import json
import subprocess
#print("The start time is :",starttime)


filename = 'C:/Users/Sabrina/Desktop/DLC Res/crayfish/otherVideos/2crayfish.mp4'

def func(video_path):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_path]).decode()
    f_types = out.replace('pict_type=','').split()
    return [i for i, type_ in enumerate(f_types) if type_ == 'I']

index = func(filename)