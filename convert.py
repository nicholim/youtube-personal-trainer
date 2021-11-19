import os

def convert_to_h264(out_path):
    out_h264 = out_path[:-7]+".mp4"
    os.system("ffmpeg -i "+out_path+" -vcodec libx264 "+out_h264)   
    return out_h264 