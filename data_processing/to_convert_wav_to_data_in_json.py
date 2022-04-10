import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os, json, math


DATASET_PATH = 'data/genres/'
JSON_PATH = "data/data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=20, n_fft=2048, hop_length=512, num_segments=5,with_spectro=False):
    if with_spectro:
        data = {
            "mapping": [],
            "labels": [],
            "mfcc": [],
            "spectro": []
        }
    else:
        data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
        }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)


    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        #print(i, dirpath, dirnames, filenames)
        if dirpath is not dataset_path:

            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            
            for f in filenames:
                
                file_path = os.path.join(dirpath, f)
                if ("hiphop.00032.wav" not in file_path and  "country.00007.wav" not in file_path and  "blues.00007.wav" not in file_path and "classical.00007.wav" not in file_path and  "disco.00007.wav"  not in file_path and  "jazz.00007.wav" not in file_path and "metal.00007.wav" not in file_path and "pop.00007.wav" not in file_path and "reggae.00007.wav" not in file_path and "rock.00007.wav" not in file_path):
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment
                        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        #mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, win_length=int(0.025*sample_rate),hop_length=int(0.01*sample_rate))
                        mfcc = mfcc.T
                        X = librosa.stft(signal[start:finish],n_fft=int(n_fft/4-1))
                        Xdb = librosa.amplitude_to_db(abs(X))
                        
                        if len(mfcc) == num_mfcc_vectors_per_segment+1:
                            data["mfcc"].append(mfcc.tolist())
                            if with_spectro:
                                data["spectro"].append(Xdb.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment:{}".format(file_path, d+1))
                        else :
                            print("{},trop court segment:{}".format(file_path, d+1))

    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
  
#data/genres/hiphop\hiphop.00032.wav,trop court segment:6
#data/genres/country\country.00007.wav,trop court segment:6
save_mfcc(DATASET_PATH,JSON_PATH , num_segments=6)
