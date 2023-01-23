from tqdm import tqdm
import os
import cv2 as cv
import pandas as pd
import numpy as np
import pickle
from config import * 

def read_train_data_paths(classes, train_path):
    data_paths = []
    truth = []
  
    for class_name in classes:
        print(class_name)
        for file in tqdm(os.listdir(train_path + "/" + class_name)):
            vid_path = os.path.join(train_path + "/" + class_name, file)
            print(vid_path)
            data_paths.append(vid_path)
            truth.append(classes[class_name])
    
    return data_paths, truth

def read_test_data_paths():
    data_paths = []

    for file in tqdm(os.listdir(test_path)):
        vid_path = os.path.join(test_path, file)
        data_paths.append(vid_path)
    
    return data_paths

def get_test_data_names():
    data_names = []
    for file in tqdm(os.listdir(test_path)):
        data_names.append(file)
    
    return data_names

def know_about_train_data(data_paths):
    vids_shape = []
    for vid_path in tqdm(data_paths):
        cap = cv.VideoCapture(vid_path)

        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height= cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        depth = cap.get(cv.CAP_PROP_FRAME_COUNT)
        
        vids_shape.append((width, height, depth))

    print("\nWe have", len(vids_shape), " videos.")

    data_counter = {}

    for class_name in classes:
            data_counter[class_name] = len(os.listdir(train_path + "/" + class_name))

    print("Videos are divided into ", data_counter)
    
    print("Videos Shapes are:")
    data_frame = pd.Series(vids_shape).value_counts()
    print(data_frame.head(len(data_frame)))

def know_about_test_data(data_paths):
        
    vids_shape = []
    for vid_path in tqdm(data_paths):
        cap = cv.VideoCapture(vid_path)

        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height= cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        depth = cap.get(cv.CAP_PROP_FRAME_COUNT)
        
        vids_shape.append((width, height, depth))
    
    print("\nWe have", len(vids_shape), " videos.")
    
    print("Videos Shapes are:")
    data_frame = pd.Series(vids_shape).value_counts()
    print(data_frame.head(len(data_frame)))

# **PREPROCESSING DATA**
sample_shape = (D, W, H, C) #Single Video shape.

def preprocess(data_paths, data_truth):
    all_videos = []

    for i in tqdm(range(len(data_paths))):
        cap = cv.VideoCapture(data_paths[i])

        single_video_frames = []
        while (True):
            read_success, current_frame = cap.read()
            
            if not read_success:
                break

            current_frame = cv.resize(current_frame, (W, H))
            single_video_frames.append(current_frame)

        cap.release()

        single_video_frames = np.array(single_video_frames)
        single_video_frames.resize((D,W,H,C), refcheck=False)

        all_videos.append(single_video_frames)
    
    all_videos = np.array(all_videos)
    data_truth = np.array(data_truth)

    return all_videos, data_truth

def save_structure(structure, name):
    with open(code_base_directory + 'Run-time/' + name + '.pickle', 'wb') as handle:
        pickle.dump(structure, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_structure(name):
    if os.path.isfile(code_base_directory + 'Run-time/' + name + '.pickle'):
        with open(code_base_directory + 'Run-time/' + name + '.pickle', 'rb') as handle:
            structure = pickle.load(handle)
        return structure
    else:
        return []