import os
import json
import cv2
import random
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, datasets
from PIL import Image
from multiprocessing import Process


def frames_from_video_file(video_path, frame_start, frame_end, n_frames, output_size = (32, 32), frame_step = 10):

    frames = []
    src = cv2.VideoCapture(video_path)

    while src.isOpened():

        if frame_end == -1:
            frame_end = cv2.CAP_PROP_FRAME_COUNT - 1

        video_len = frame_end - frame_start
        need_len = 1 + (n_frames - 1) * frame_step

        if need_len > video_len:
            start = frame_start
        else:
            start = random.randint(frame_start, video_len - need_len + frame_start + 1)

        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        success, frame = src.read()

        if not success:
            break

        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                success, frame = src.read()
            if success:
                frame = format_frame(frame, output_size)
                frames.append(frame)
            else:
                frames.append(np.zeros_like(frames[0]))

        src.release()

        frames = np.array(frames)[..., [2, 1, 0]] # Reorder the array bc tensor images and images are stored weird

    return frames

def format_frame(frame, output_size):

    frame = frame.astype(np.float32)
    frame = torch.from_numpy(frame)

    transform = transforms.Compose([
        transforms.Resize(output_size),
    ])

    return transform(frame)

def instance_videos(gloss, instances, split):
    # Make sure we have directories to put things

    if not os.path.exists("../asl_video_frames/"):
        os.mkdir("../asl_video_frames")
    if not os.path.exists("../asl_video_frames/" + split):
        os.mkdir("../asl_video_frames/" + split)
    if not os.path.exists("../asl_video_frames/" + split + "/" + gloss):
        os.mkdir("../asl_video_frames/" + split + "/" + gloss)

    for instance in instances:

        if instance["split"] != split:
            continue

        # vid_path = instance["url"]
        vid_path = "../asl_video/videos/" + instance["video_id"] + ".mp4"
        start = instance["frame_start"]
        end = instance["frame_end"]
        frames = frames_from_video_file(vid_path, start, end, 16)

        if len(frames) == 0:
            continue

        if not os.path.exists("../asl_video_frames/" + split + "/" + gloss + "/" + str(instance["instance_id"])):
            os.mkdir("../asl_video_frames/" + split + "/" + gloss + "/" + str(instance["instance_id"]))
        
        wd = os.getcwd()
        os.chdir("../asl_video_frames/" + split + "/" + gloss + "/" + str(instance["instance_id"]))
        for i in range(len(frames)):
            cv2.imwrite(str(i) + ".jpg", frames[i], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        os.chdir(wd)

def extract(split):

    with open("./WLASL_v0.3.json") as file: 
        data = json.load(file)
    for datum in data:
        instance_videos(datum["gloss"], datum["instances"], split)

def extract_all():

    if __name__ == '__main__':
        train = Process(target = extract, args = ("train",))
        test = Process(target = extract, args = ("test",))
        val = Process(target = extract, args = ("val",))
        train.start()
        test.start()
        val.start()
        train.join()
        test.join()
        val.join()

extract_all()