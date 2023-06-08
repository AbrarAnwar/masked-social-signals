import numpy as np
import json
data_dir = 'dining_dataset/upsampled-person-speaking/'
words_dir = 'dining_dataset/processed_audio/v1/'
target_dir = 'dining_dataset/words/v1/'


def seconds_to_frame(seconds):
    return int(seconds * 30)


def data_load(status, words_dir):
    data = np.load(status)
    words = json.load(open(words_dir))
    return data, words


def get_frame_range(item):
    frame_range = (seconds_to_frame(item['start']), seconds_to_frame(item['end'])+1)
    frame_index = np.arange(frame_range[0], frame_range[1])

    return frame_index

def find_recent_speaker(data, frame_index):
    prev, nex = frame_index[0], frame_index[-1]

    while prev > 0 and data[prev] == 0:
        prev -= 1

    while nex < len(data) - 1 and data[nex] == 0:
        nex += 1

    prev_dist = frame_index[0] - prev
    nex_dist = nex - frame_index[-1]
    speaker = None

    if prev_dist !=0 and nex_dist !=0:
        if prev_dist < nex_dist:
            speaker = data[prev]
        else:
            speaker = data[nex]
        
    elif prev_dist == 0:
        speaker = data[nex]
    elif nex_dist == 0:
        speaker = data[prev]

    return speaker

def assign_words(data, frame_index, window=5):
    frame_seg = data[frame_index]

    if sum(frame_seg) == 0:
        #print('no speaker')
        return find_recent_speaker(data, frame_index)

    else:
        unique_values, counts = np.unique(frame_seg[frame_seg != 0], return_counts=True)

        max_count = np.max(counts)
        num_max_counts = np.sum(counts == max_count)

        if num_max_counts > 1:
            #print('tie')
            max_indices = np.argsort(counts)[::-1][:num_max_counts]
            speakers = unique_values[max_indices]

            for i in frame_seg:
                if i in speakers:
                    speaker = i
                    break
        else:
            max_count_index = np.argmax(counts)
            speaker = unique_values[max_count_index]

    return speaker

if __name__ == '__main__':
    for i in range(30):
        print(i+1)
        if i+1 == 9:
            continue

        status = data_dir + '{:02d}.npy'.format(i+1)
        words = words_dir + '{:02d}.json'.format(i+1)
        output = target_dir + '{:02d}.jsonl'.format(i+1)
        if i+1 == 3:
            words = 'dining_dataset/processed_audio/v1_noVAD/' + '{:02d}.json'.format(i+1)
        
        data, words = data_load(status, words)
        json_data = [{'id': str(j), 'status_speaker': str(s), 'whisper_speaker': str(s), 'words': str(-s)} for j, s in enumerate(data)]

        
        for sentence in words['segments']:
            for item in sentence['words']:
                word = item['text']
                frame_index = get_frame_range(item)
                
                if frame_index[0] >= len(data):
                    print('severe:', i+1)
                    break
                
                if len(data) in frame_index:
                    print('simple:', i+1)
                    frame_index = frame_index[:np.where(frame_index == len(data))[0][0]]
                
                speaker = assign_words(data, frame_index)

                for j in frame_index:
                    if json_data[j]['words'] in ['0','-1','-2','-3']:
                        json_data[j]['words'] = word
                    json_data[j]['whisper_speaker'] = str(speaker) 
                
        
        with open(output, 'w') as f:
            for i in json_data:
                json_line = json.dumps(i)
                f.write(json_line + '\n') 



