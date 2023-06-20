import json
import pathlib
import csv
import numpy as np
from scipy.interpolate import interp1d

word_dir = 'dining_dataset/words/v1/'
gaze_dir = 'dining_dataset/processed_gazes/'
status_dir = 'dining_dataset/upsampled-person-speaking/'
keypoints_dir = 'vision_openpose_features/'

process_gazepose_dir = 'dining_dataset/full_gazes/'
process_keypoints_dir = 'dining_dataset/full_keypoints/'
clean_keypoints_dir = 'dining_dataset/clean_keypoints/'

def calculate_length():
    result = []

    for i in range(30):
        if i+1 == 9:
            result.append(0)
            continue

        status_path = status_dir + '{:02d}'.format(i+1) + '.npy'
        status_array = np.load(status_path)

        word_path = word_dir + '{:02d}'.format(i+1) + '.jsonl'
        

        with open(word_path, 'r') as f:
            for word_length, _ in enumerate(f, start=1):
                pass

        assert len(status_array) == word_length, 'File: {} status length: {}, word length: {}'.format(status_path, len(status_array), word_length)

        result.append(len(status_array))

    return result


def check_length(length_list, is_keypoint=True):
    print('Checking...')
    count = 0
    for i in range(30):
        if i+1 == 9:
            continue
        frame_length = length_list[i]
        
        if is_keypoint:
            for person in range(3):
                person_dir = pathlib.Path(keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '/')
                for keypoint_length, file in enumerate(sorted(person_dir.iterdir()), start=1):
                    pass
            
                if keypoint_length != frame_length:
                    count += 1
                    print('File ({}) inconsistent, word_length: {}, keypoint_length: {}'.format(person_dir, frame_length, keypoint_length))
        
        else:
            for person in range(3):
                gaze_path = gaze_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.csv'
                with open(gaze_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for gaze_length, row in enumerate(reader, start=1):
                        pass
                        
                if frame_length != gaze_length:
                    count += 1
                    print('File ({}) inconsistent, word_length: {}, gaze_length: {}'.format(gaze_path, frame_length, gaze_length))
        
    print('total inconsistent file:', count)


def process_gazepose(length_list):
    print('Processing gazepose...')
    for i in range(30):
        if i+1 == 9:
            continue

        frame_length = length_list[i]

        for person in range(3):
            headpose_list = np.zeros((frame_length, 2))
            gaze_list = np.zeros((frame_length, 2))
            gaze_path = gaze_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.csv'
            print(gaze_path)

            with open(gaze_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    index = int(row['name'])
                    headpose = row['headpose']
                    gaze = row['gaze']

                    if index < frame_length:
                        headpose_list[index] = np.array(eval(headpose))
                        gaze_list[index] = np.array(eval(gaze))
            #count = 0
            for index in range(frame_length):
                if headpose_list[index][0] == 0 and headpose_list[index][1] == 0:
                    #count += 1
                    #print(index)
                    if index == 0:
                        replace = index + 1
                        while headpose_list[replace][0] == 0 and headpose_list[replace][1] == 0:
                            replace += 1

                        headpose_list[index] = headpose_list[replace]
                        gaze_list[index] = gaze_list[replace]
                    else:
                        replace = index - 1

                        headpose_list[index] = headpose_list[replace]
                        gaze_list[index] = gaze_list[replace]

            assert not any((headpose_list == np.zeros(2)).all(1)) and \
                not any((gaze_list == np.zeros(2)).all(1)), 'File: {} inconsistent, empty file not handled'.format(gaze_path)
            #print('total empty:', count)
            #print('start writing')

            process_path = process_gazepose_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'

            np.savez(process_path, headpose=np.array(headpose_list), gaze=np.array(gaze_list))

                    

def process_keypoints(length_list):
    print('Processing keypoints...')
    for i in range(30):
        if i+1 == 9:
            continue

        frame_length = length_list[i]
        

        for person in range(3):
            pose_list = np.zeros((frame_length, 75))
            #face_list = np.zeros((frame_length, 210))
            empty = []

            person_dir = pathlib.Path(keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '/')
            print(person_dir)

            for keypoint_length, file in enumerate(sorted(person_dir.iterdir()), start=0):
                assert keypoint_length == int(file.name.split('_')[2]), 'Missing file: {} {}'.format(keypoint_length, person+1)
                
                with open(file, 'r') as f:
                    keypoints = json.load(f)['people']


                if keypoints != []:
                    pose = keypoints[0]['pose_keypoints_2d']
                    face = keypoints[0]['face_keypoints_2d']
                    
                    if pose == []:
                        print('pose empty')
                        print(file)
                    else:
                        if keypoint_length < frame_length:
                            pose_list[keypoint_length] = np.array(pose)

                    if face != []:
                        print('face not empty')
                        print(file)
            
            # handle empty files
            for index in range(frame_length):
                if sum(pose_list[index]) == 0:
                    if index == 0:
                        replace = index + 1
                        while sum(pose_list[replace]) == 0:
                            replace += 1

                        pose_list[index] = pose_list[replace]
                        #face_list[index] = face_list[replace]
                    else:
                        replace = index - 1

                        pose_list[index] = pose_list[replace]
                        #face_list[index] = face_list[replace]
            
            assert not any((pose_list == np.zeros(75)).all(1)), 'File: {} inconsistent, empty file not handled'.format(person_dir)
            #print('start writing')

            # write to jsonline file for each person and each video
            process_path = process_keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'

            np.savez(process_path, pose=pose_list)
            


def index_pose(array):
     # 9, 10, 11, 12
    indices = [0,1,2,3,4,5,6,7,8,15,16,17,18]
    indices_xy = [j for i in indices for j in (i*3, i*3 + 1)] 

    return array[:, indices_xy]



def clean_pose():
    # read npz file
    for i in range(30):
        if i+1 == 9:
            continue

        for person in range(3):
            file_name = process_keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'
            print('Cleaning file: {}'.format(file_name))
            pose = np.load(file_name)['pose']
            new_pose = index_pose(pose)
            

            for column in range(0, new_pose.shape[-1]):
                valid_entries = np.nonzero(new_pose[:, column])[0]
                missing_entries = np.where(new_pose[:, column] == 0)[0]

                interp_func = interp1d(valid_entries, new_pose[valid_entries, column], bounds_error=False)

                new_pose[missing_entries, column] = interp_func(missing_entries)
                
                first_non_zero = new_pose[valid_entries[0], column]
                last_non_zero = new_pose[valid_entries[-1], column]
                new_pose[:valid_entries[0]] = first_non_zero
                new_pose[valid_entries[-1] + 1:] = last_non_zero
                
                assert not any((new_pose[:, column] == 0)), 'File: {} column: {} has 0'.format(file_name, column)
                
            # write to jsonline file for each person and each video
            clean_file = clean_keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'
            np.savez(clean_file, pose=new_pose)

            



if __name__ == '__main__':
    #total_length = calculate_length()
    #check_length(total_length, True)
    #process_gazepose(total_length)
    #process_keypoints(total_length)
    clean_pose()
    

