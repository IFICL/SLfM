import os
import shutil
import numpy as np
import glob
import argparse
import random
import json
from tqdm import tqdm
import csv
import cv2
import soundfile as sf
from PIL import Image
import PIL
from scipy.io import wavfile
from scipy.signal import fftconvolve
from pyroomacoustics.experimental.rt60 import measure_rt60

from util import sound

parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--dataset', default='', type=str)
parser.add_argument('--delete', default=False, action='store_true')
parser.add_argument('--process_meta', default=False, action='store_true')
parser.add_argument('--downsample', default=False, action='store_true')
parser.add_argument('--render_audio', default=False, action='store_true')
parser.add_argument('--mix_reverb', default=False, action='store_true')
parser.add_argument('--calc_reverb', default=False, action='store_true')
parser.add_argument('--indirect_ratio', type=float, default=0.25)

parser.add_argument('--split', type=int, default=0, help='i split of data to process')
parser.add_argument('--total', type=int, default=1, help='total splits')

random.seed(1234)


def sample_audio_database():
    np.random.seed(1234)
    database = glob.glob('LibriSpeech/ProcessedData/*/*/*/audio.wav')
    database.sort()
    sampled_audio_paths = np.random.choice(database, 1, replace=False)
    source_sounds = []
    for audio_path in sampled_audio_paths:
        source_sound, _ = sf.read(audio_path, start=0, stop=int(
            16000 * 10), dtype='float32', always_2d=True)
        source_sound = source_sound.mean(-1)
        desired_rms = 0.03 * np.random.rand() + 0.07
        # source_sound = normalize_audio(source_sound, desired_rms=desired_rms)
        source_sounds.append(source_sound)
    return source_sounds


def impulse_response_to_sound(binaural_rir, source_sound, sampling_rate):
    '''
        goal: create sound based on simulate impulse response
        binaural_rir: (num_sample, num_channel)
        source_sound: mono sound, (num_sample)
        rir and source sound should have same sampling rate
    '''
    binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel]) for channel in range(binaural_rir.shape[-1])])
    return binaural_convolved


def postprocess_meta(args, pair_list):
    for pair in tqdm(pair_list, desc=f'Processing ID = {str(args.split).zfill(2)}'):
        meta_path = os.path.join(pair, 'metadata.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)

        n_source = np.array([key.find('source') != -1 and key.find('position') != -1 for key in meta_dict.keys()]).sum()
        n_camera = np.array([key.find('camera') != -1 and key.find('position') != -1 for key in meta_dict.keys()]).sum()

        if not args.delete:
            for source_id in range(n_source):
                for camera_id in range(n_camera):
                    pos_sound = np.array(meta_dict[f'source_{source_id}_position'])
                    pos_agent = np.array(meta_dict[f'camera_{camera_id}_position'])
                    angle_agent = meta_dict[f'camera_{camera_id}_angle']
                    angle = sound.calc_sound_direction_for_agent( pos_sound, pos_agent, angle_agent)
                    meta_dict[f'relative_angle_between_sound_{source_id}_camera_{camera_id}'] = angle
        else:
            key_list = meta_dict.copy().keys()
            for key in key_list:
                if key.find('relative_angle') != -1:
                    meta_dict.pop(key)
        with open(meta_path, 'w') as fo:
            json.dump(meta_dict, fo, indent=4, sort_keys=False)


def postprocess_downsample(args, pair_list):
    for pair in tqdm(pair_list, desc=f'Processing ID = {str(args.split).zfill(2)}'):
        meta_path = os.path.join(pair, 'metadata.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
        n_camera = np.array([key.find('camera') != -1 and key.find('position') != -1 for key in meta_dict.keys()]).sum()
        for camera_id in range(n_camera):
            # import pdb; pdb.set_trace()
            img_path = os.path.join(pair, 'high_resol', f'camera_{camera_id}_rgb.png')
            if not os.path.exists(img_path):
                print(f'Missing {img_path}')
                exit()
            
            save_path = os.path.join(pair, f'camera_{camera_id}_rgb.png')
            if args.delete:
                if os.path.exists(save_path):
                    os.remove(save_path)
            else:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((320, 240), resample=PIL.Image.Resampling.BILINEAR)
                img.save(save_path)


def postprocess_render_audio(args, pair_list):
    for pair in tqdm(pair_list, desc=f'Processing ID = {str(args.split).zfill(2)}'):
        render_path = os.path.join(pair, 'render_audios')
        if args.delete:
            if os.path.exists(render_path):
                shutil.rmtree(render_path)


def postprocess_mix_direct_and_indirect_rir(args, pair_list):
    for pair in tqdm(pair_list, desc=f'Processing ID = {str(args.split).zfill(2)}'):
        direct_rirs = glob.glob(f"{pair}/binaural_rirs_direct/*")
        indirect_rirs = glob.glob(f"{pair}/binaural_rirs_indirect/*")
        direct_rirs.sort()
        indirect_rirs.sort()
        save_folder = f"{pair}/binaural_rirs"

        if args.delete:
            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)
            continue

        os.makedirs(save_folder, exist_ok=True)

        # audio_save_folder = f"{pair}/render_audios"
        # if os.path.exists(audio_save_folder):
        #     shutil.rmtree(audio_save_folder)

        meta_path = os.path.join(pair, 'metadata.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)

        for i in range(len(direct_rirs)):
            # import pdb; pdb.set_trace()
            direct_rir, rate = sf.read(direct_rirs[i], dtype='float32', always_2d=True)
            indirect_rir, _  = sf.read(indirect_rirs[i], dtype='float32', always_2d=True)
            zero_padding = np.zeros((indirect_rir.shape[0] - direct_rir.shape[0], direct_rir.shape[1]))
            padded_direct_rir = np.concatenate((direct_rir, zero_padding), axis=0)
            scaling = args.indirect_ratio
            scaled_indirect_rir = scaling * indirect_rir
            rir = padded_direct_rir + scaled_indirect_rir
            save_path = os.path.join(save_folder, direct_rirs[i].split('/')[-1])

            sf.write(save_path, rir, rate)

            rt60_l = measure_rt60(rir[:, 0], rate, decay_db=30, plot=False)
            rt60_r = measure_rt60(rir[:, 1], rate, decay_db=30, plot=False)

            meta_dict[f"{direct_rirs[i].split('/')[-1][:-4]}_rt60"] = [rt60_l, rt60_r]


        with open(meta_path, 'w') as fo:
            json.dump(meta_dict, fo, indent=4, sort_keys=True)


def calcualte_average_reverb(args, pair_list):
    rt60s = []
    for pair in tqdm(pair_list, desc=f'Processing ID = {str(args.split).zfill(2)}'):
        direct_rirs = glob.glob(f"{pair}/binaural_rirs_direct/*")
        indirect_rirs = glob.glob(f"{pair}/binaural_rirs_indirect/*")
        direct_rirs.sort()
        indirect_rirs.sort()

        for i in range(len(direct_rirs)):
            # import pdb; pdb.set_trace()
            direct_rir, rate = sf.read(direct_rirs[i], dtype='float32', always_2d=True)
            indirect_rir, _ = sf.read(indirect_rirs[i], dtype='float32', always_2d=True)
            zero_padding = np.zeros((indirect_rir.shape[0] - direct_rir.shape[0], direct_rir.shape[1]))
            padded_direct_rir = np.concatenate((direct_rir, zero_padding), axis=0)
            scaled_indirect_rir =  args.indirect_ratio * indirect_rir
            rir = padded_direct_rir + scaled_indirect_rir

            rt60_l = measure_rt60(rir[:, 0], rate, decay_db=30, plot=False)
            rt60_r = measure_rt60(rir[:, 1], rate, decay_db=30, plot=False)

            rt60s.append(np.array([rt60_l, rt60_r]))
            # import pdb; pdb.set_trace()

    rt60s = np.concatenate(rt60s, axis=0).mean()
    print(rt60s)




def main(args):
    # import pdb; pdb.set_trace()
    read_path = f'ProcessedData/{args.dataset}'
    data_list = glob.glob(f'{read_path}/*/*')
    data_list.sort()

    data_list = data_list[int(args.split / args.total * len(data_list)): int((args.split+1) / args.total * len(data_list))]

    if args.process_meta:
        postprocess_meta(args, data_list)

    if args.downsample:
        postprocess_downsample(args, data_list)

    if args.render_audio:
        postprocess_render_audio(args, data_list)

    if args.mix_reverb:
        postprocess_mix_direct_and_indirect_rir(args, data_list)
    
    if args.calc_reverb:
        calcualte_average_reverb(args, data_list)



# Usage: python postprocess.py --dataset='hm3d-rotation-noreverb-v2'
# Usage: python postprocess.py --dataset='hm3d-rotation-noreverb-v2' --downsample
# Usage: python postprocess.py --dataset='hm3d-3view-noreverb' --downsample
# Usage: python postprocess.py --dataset='hm3d-3view-noreverb' --render_audio --delete
# Usage: python postprocess.py --dataset='human-test' --mix_reverb
# Usage: python postprocess.py --dataset='hm3d-4view-rotation' --calc_reverb --indirect_ratio=0.2   --split=0 --total=10
# Usage: python postprocess.py --dataset='hm3d-4view-rotation' --mix_reverb  --indirect_ratio=0.2  --split=0 --total=20


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
