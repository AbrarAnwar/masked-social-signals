import json
from pathlib import Path
import csv
import numpy as np
from words import seconds_to_frame

word_dir = 'dining_dataset/words/v1/'
gaze_dir = 'dining_dataset/full_gazes/'
status_dir = 'dining_dataset/upsampled-person-speaking/'
keypoints_dir = 'dining_dataset/clean_keypoints/'

# write to jsonline file 9k lines
# {id , word, status_speaker, whisper_speaker, gaze:, headpose, body, face}
#
#
#  id , word, status_speaker, whisper_speaker, in words folder on each video
#   gaze:, headpose in each video and each person
#   body, face in each video and each person

def read_word():

    for i in range(30):
        if i+1 == 9:
            continue
        word_path = word_dir + '{:02d}'.format(i+1) + '.jsonl'

        with open(word_path, 'r') as f:
            words = [json.loads(line) for line in f]

        yield i, words


def read_gazepose():

    for i in range(30):
        result = []

        if i+1 == 9:
            continue
        
        for person in range(3):
            gaze_path = gaze_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'
            npzfile = np.load(gaze_path)
            headpose = npzfile['headpose']
            gaze = npzfile['gaze']
            result.append((headpose, gaze))

        yield result
        
     

def read_keypoint():

    for i in range(30):
        result = []
        if i+1 == 9:
            continue
        for person in range(3):
            keypoint_path = keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'
            npzfile = np.load(keypoint_path)
            pose = npzfile['pose']
            result.append(pose)

        yield result

def write_to_batch(window_size, stride_size, is_second=True):
    count = 0
    window, stride = (seconds_to_frame(window_size), seconds_to_frame(stride_size)) if is_second else (window_size, stride_size)

    # create directory for batch
    target_dir = 'dining_dataset/batch_window{}_stride{}/'.format(window_size, stride_size)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for (idx, words), headpose_gaze, pose in zip(read_word(), read_gazepose(), read_keypoint()):
        print(idx+1)

        assert len(words) == headpose_gaze[0][0].shape[0] == headpose_gaze[1][0].shape[0] == headpose_gaze[2][0].shape[0] == pose[0].shape[0] == pose[1].shape[0] == pose[2].shape[0], \
            'Inconsistent! word_length: {}, headpose_length: {}, pose_length: {}'.format(len(words), headpose_gaze[0][0].shape[0], pose[0].shape[0])

        start = 0
        #print('start')
        while start < len(words) - stride:
            
            end = start + window
            
            if end > len(words):
                end = len(words)
                if end - start < window:
                    start = end - window
            each_person = [{'word':[], 'status_speaker': [],'whisper_speaker': [], 'headpose': [], 'gaze': [], 'pose': []} for _ in range(3)]
            
            # word
            word_segment = words[start:end]
            all_word = [['' for _ in range(window)] for _ in range(4)]
            status = np.zeros((4, window))
            whisper = np.zeros((4, window))
            
            for i, w in enumerate(word_segment):
                #print(w['whisper_speaker'])
                all_word[int(w['whisper_speaker'])][i] = w['words']
                status[int(w['status_speaker']), i] = 1
                whisper[int(w['whisper_speaker']), i] = 1
                
            for i in range(3):
                each_person[i]['word'] = np.array(all_word[i+1])
                each_person[i]['status_speaker'] = status[i+1]
                each_person[i]['whisper_speaker'] = whisper[i+1]
            
            
            # headpose, gaze, and pose
            headpose_segment = [headpose[start:end] for headpose, _ in headpose_gaze]
            gaze_segment = [gaze[start:end] for _, gaze in headpose_gaze]
            pose_segment = [pose[start:end] for pose in pose]

            for i in range(3):
                each_person[i]['headpose'] = np.array(headpose_segment[i])
                each_person[i]['gaze'] = np.array(gaze_segment[i])
                each_person[i]['pose'] = np.array(pose_segment[i])
            
            # save to npz file
            save_dict = {}
            target_path = target_dir + '{:d}.npz'.format(count)
            for i, person_data in enumerate(each_person):
                for key, array in person_data.items():
                    save_dict[f'person_{i}_{key}'] = array

            np.savez(target_path, **save_dict)

            start += stride
            count += 1
    

if '__main__' == __name__:  
    write_to_batch(36, 18)