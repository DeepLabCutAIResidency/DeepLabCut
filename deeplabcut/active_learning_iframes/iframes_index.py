# %%
import time
import random
import json
import subprocess
# %%
start_time = time.time()

filename = '/media/data/crayfish/otherVideos/2crayfish.mp4'


start_time = time.time()

def get_frames_metadata(file):
    command = '"{ffexec}" -show_frames -print_format json "{filename}"'.format(ffexec='ffprobe', filename=file)
    response_json = subprocess.check_output(command, shell=True, stderr=None)
    frames = json.loads(response_json)["frames"]
    frames_metadata, frames_type, frames_type_bool = [], [], []
    for frame in frames:
        if frame["media_type"] == "video":
            video_frame = json.dumps(dict(frame), indent=4)
            frames_metadata.append(video_frame)
            frames_type.append(frame["pict_type"])
            if frame["pict_type"] == "I":
                frames_type_bool.append(True)
            else:
                frames_type_bool.append(False)
    #print(frames_type)
    return frames_type

frames_type = get_frames_metadata(filename)

print("--- %s seconds ---" % (time.time() - start_time))


# %%
index_i= []
for i in range(len(frames_type)):
    if frames_type[i] == 'I':
        index_i.append(i)

frames2pick = 20
frames2label = random.sample(index_i, frames2pick)
print(frames2label) #index to label


# %%

##fast code:

#if (filename.endswith(".mp4")): #or .avi, .mpeg, whatever.
#    os.system('ffmpeg -skip_frame nokey -i "' + str(filename) + '" -vsync vfr -frame_pts true out-%02d.jpeg')


#print("--- %s seconds ---" % (time.time() - start_time))