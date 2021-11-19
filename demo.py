from DTWdistance import distance_using_dtw
from extract_from_user_video import output_user_keypoints
from youtube_links import links
import argparse
import glob, os

parser = argparse.ArgumentParser( description='Rank results of user video compared to database')
parser.add_argument('--video', help='User video name in .mp4 format (write with .mp4)')     
parser.add_argument('--model', help='body, coco, or mpi')  
args = parser.parse_args()  

directory = os.getcwd()
user_video_path = directory+"/"+args.video

if args.model=="body":
    user_npy_path_body = directory+"/"+args.video[:len(args.video)-4]+"_body.npy"
    output_user_keypoints(user_video_path, user_npy_path_body, 0.2, model=args.model)
    rank_result = distance_using_dtw(args.model, user_npy_path_body)
    rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1]))
    print("Here are some suggested videos for you to watch in YouTube \n")
    for k,v in rank_result.items():
        # print k[:-9]
        print (k[:-9] + ' with value ' + str(v))
        print (links[k[:-9]]+'\n')

elif args.model=="coco":
    user_npy_path_coco = directory+"/"+args.video[:len(args.video)-4]+"_coco.npy"
    output_user_keypoints(user_video_path, user_npy_path_coco, 0.2, model=args.model)
    rank_result = distance_using_dtw(args.model, user_npy_path_coco)
    rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1]))
    print("Here are some suggested videos for you to watch in YouTube \n")
    for k,v in rank_result.items():
        # print k[:-9]
        print (k[:-9] + ' with value ' + str(v))
        print (links[k[:-9]]+'\n')

elif args.model=="mpi":
    user_npy_path_mpi = directory+"/"+args.video[:len(args.video)-4]+"_mpii.npy"
    output_user_keypoints(user_video_path, user_npy_path_mpi, 0.2, model=args.model)
    rank_result = distance_using_dtw(args.model, user_npy_path_mpi)
    rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1]))
    print("Here are some suggested videos for you to watch in YouTube \n")
    for k,v in rank_result.items():
        # print k[:-9]
        print (k[:-9] + ' with value ' + str(v))
        print (links[k[:-9]]+'\n')

else:
    print("please choose the model either body, coco, or mpi")