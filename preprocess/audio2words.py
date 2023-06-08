from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import os
os.environ['TIKTOKEN_CACHE_DIR'] = '/home/tangyimi/tmp/'


import whisper_timestamped as whisper
import json

model = whisper.load_model("large-v1", device="cuda")

data_path = 'dining_dataset/audio/'
target_path = 'dining_dataset/processed_audio_v1_noVAD/'

if __name__ == '__main__':
    for i in range(30):
        if i+1 == 9:
            continue
        file_path = data_path + '{:02d}.wav'.format(i+1)
        print(file_path)

        audio = whisper.load_audio(file_path)

        result = whisper.transcribe(model, audio, language="en")

        json_object = json.dumps(result, indent = 2, ensure_ascii = False)

        with open(target_path + '{:02d}.json'.format(i+1), "w") as outfile:
            outfile.write(json_object)
