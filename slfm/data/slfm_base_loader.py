import csv
import glob
import io
import json
import librosa
import numpy as np
import os
import pickle
from PIL import Image
from PIL import ImageFilter
import random
import scipy
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import fftconvolve
import time
from tqdm import tqdm
import glob
import cv2

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from utils import sound, sourcesep
from data import * 



class SLfMbaseDataset(object):
    def __init__(self, args, pr, list_sample, split='train'):
        self.pr = pr
        self.args = args
        self.split = split
        self.seed = pr.seed
        self.online_render = args.online_render
        self.time_sync = args.time_sync
        self.not_load_audio = args.not_load_audio
        self.n_view = args.n_view
        # save args parameter
        self.repeat = args.repeat if split == 'train' else 1
        self.max_sample = args.max_sample if split in ['train', 'test'] else -1

        self.image_transform = transforms.Compose(self.generate_image_transform(args, pr))

        self.list_sample = self.get_list_sample(list_sample)
        if self.max_sample > 0: 
            self.list_sample = self.list_sample[0:self.max_sample]

        self.list_sample = self.list_sample * self.repeat

        # init random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Random Generator
        self.rng = np.random.default_rng(self.seed)

        num_sample = len(self.list_sample)
        if self.split == 'train':
            random.shuffle(self.list_sample)
        
        if self.online_render:
            self.audio_database = self.generate_audio_database()

        # import pdb; pdb.set_trace()
        # always load training relative angle distribution for angle bins
        self.angle_distribution = self.relative_angle_distribution()
        self.source_distribution = self.source_direction_distribution()

        print('Audio Dataloader: # sample of {}: {}'.format(self.split, num_sample))


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        pair_path = info['path']
        meta_path = os.path.join(pair_path, 'metadata.json')
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)

        max_camera_num = np.array([key.find('camera') != -1 and key.find('position') != -1 for key in meta_dict.keys()]).sum()
        if self.n_view > max_camera_num:
            print("camera numbers are not match")
            raise NotImplementedError

        img_list = []
        img_path_list = []
        depth_path_list = []
        audio_list = []
        camera_angle_list = []
        camera_posit_list = []
        source_direction_list = []

        if self.online_render:
            source_sounds = self.prepare_source_sounds()
        else:
            source_sounds = None
        
        for camera_ind in range(self.n_view):
            img_path = os.path.join(pair_path, f'camera_{camera_ind}_rgb.png')
            img = self.read_image(img_path)
            img_list.append(img)
            camera_angle_list.append(meta_dict[f'camera_{camera_ind}_angle'])
            camera_posit_list.append(meta_dict[f'camera_{camera_ind}_position'])
            img_path_list.append(img_path)
            
            if self.not_load_audio:
                audio = np.zeros((2, np.rint(self.pr.clip_length * self.pr.samp_sr * 3).astype(int)))
            else:
                audio = self.generate_audio(index, pair_path, camera_ind, source_sounds)
            audio_list.append(audio)

            source_direction = float(meta_dict[f"relative_angle_between_sound_0_camera_{camera_ind}"])
            source_direction_bin = self.calc_source_direction_to_bin(source_direction)
            source_direction_list.append((source_direction, source_direction_bin))

        shuffle_flag = self.rng.random() > 0.5 if self.split == 'train' else False
        # shuffle_flag = False
        if shuffle_flag:
            shuffled_inds = self.rng.permutation(self.n_view)
            img_list = [img_list[i] for i in shuffled_inds]
            img_path_list = [img_path_list[i] for i in shuffled_inds]
            audio_list = [audio_list[i] for i in shuffled_inds]
            camera_angle_list = [camera_angle_list[i] for i in shuffled_inds]
            camera_posit_list = [camera_posit_list[i] for i in shuffled_inds]
            source_direction_list = [source_direction_list[i] for i in shuffled_inds]

        fast_step = 0.5
        slow_step = 0.05
        slow_point = self.rng.choice(int(fast_step / slow_step + 1))
        if self.time_sync:
            fast_point = self.rng.choice(np.floor(self.pr.clip_length * 2 / fast_step).astype(int))
            fast_point = np.array([fast_point] * self.n_view)
        else:
            fast_point = self.rng.choice(np.floor(self.pr.clip_length * 2 / fast_step).astype(int), self.n_view, replace=False)
        
        start_times = (fast_step * fast_point + slow_step * slow_point) * self.pr.samp_sr
        for i in range(len(audio_list)):
            audio_segment = audio_list[i][:, int(start_times[i]): int(start_times[i] + self.pr.clip_length * self.pr.samp_sr)]
            audio_list[i] = torch.tensor(audio_segment).float().unsqueeze(0)

        audio_list = torch.cat(audio_list, dim=0)

        batch = {'pair_path': pair_path}
        if self.n_view > 1:
            for i in range(1, self.n_view): 
                relative_camera_angle, relative_camera_angle_bin = self.calculate_relative_rotation([camera_angle_list[0], camera_angle_list[i]], return_bin=True)
                batch[f'relative_camera{i}_angle'] = relative_camera_angle
                batch[f'relative_camera{i}_angle_bin'] = relative_camera_angle_bin
                batch[f'relative_camera{i}_angle_sign'] = np.sign(relative_camera_angle)


        for ind in range(self.n_view):
            batch[f'img{ind+1}_path'] = img_path_list[ind]
            batch[f'img_{ind+1}'] = img_list[ind]
            batch[f'audio_{ind+1}'] = audio_list[ind]
            batch[f'angle_between_source1_camera{ind+1}'] = source_direction_list[ind][0]
            batch[f'angle_bin_between_source1_camera{ind+1}'] = source_direction_list[ind][1]
            batch[f'camera_{ind+1}_position'] = torch.tensor(camera_posit_list[ind])

        for source_ind in range(self.args.n_source):
            batch[f'source_{source_ind+1}_position'] = torch.tensor(meta_dict[f'source_{source_ind}_position'])

        return batch


    def getitem_test(self, index):
        self.__getitem__(index)


    def __len__(self): 
        return len(self.list_sample)


    def get_list_sample(self, list_sample):
        if isinstance(list_sample, str):
            samples = []
            csv_file = csv.DictReader(open(list_sample, 'r'), delimiter=',')
            for row in csv_file:
                samples.append(row)
        return samples 
    

    def generate_audio_database(self):
        audiobase_csv = f'{self.pr.audiobase_path}/{self.split}.csv'
        audio_database = self.get_list_sample(audiobase_csv)
        return audio_database


    def read_audio(self, audio_path, start=0, stop=None):
        # import pdb; pdb.set_trace()
        audio, audio_rate = sf.read(audio_path, start=start, stop=stop, dtype='float32', always_2d=True)
        # repeat in case audio is too short
        if not stop == None:
            desired_audio_length = int(stop - start)
            if audio.shape[0] < desired_audio_length:
                repeat_times = np.ceil(desired_audio_length / audio.shape[0])
                audio = np.tile(audio, (int(repeat_times), 1))[:desired_audio_length, :]

        if audio_rate != self.pr.samp_sr:
            audio = scipy.signal.resample(audio, int(audio.shape[0] / audio_rate * self.pr.samp_sr), axis=0)
            audio_rate = self.pr.samp_sr
        return audio, audio_rate
    

    def prepare_source_sounds(self):
        '''
            We preload the sound source to ensure the same sound track for each rir
            If with dominant sound option, we set the first sound source as the dominant one
        '''
        dominant_rms = None
        source_sound_paths = self.rng.choice(self.audio_database, self.args.n_source, replace=False)
        source_sounds = []
        for source_ind in range(self.args.n_source):
            source_sound_path = os.path.join(source_sound_paths[source_ind]['path'], 'audio.wav')
            with open(os.path.join(source_sound_paths[source_ind]['path'], 'meta.json'), "r") as f:
                source_meta = json.load(f)
            audio_rate = source_meta['audio_sample_rate']
            audio_length = source_meta['audio_length']
            clip_length = np.rint(self.pr.clip_length * audio_rate * 3).astype(int)
            remain_length = int(audio_length - self.pr.clip_length * 3)
            if self.split == 'train' and remain_length > 0:
                start = int(self.rng.choice(remain_length) * audio_rate)
            else:
                start = 0
            source_sound, _ = self.read_audio(source_sound_path, start=start, stop=start+clip_length)
            source_sound = source_sound.mean(-1)
            if self.args.with_dominant_sound and not self.args.ssl_flag:
                if source_ind == 0:
                    dominant_rms = desired_rms = 0.06 * self.rng.random() + 0.08
                else:
                    snr = self.rng.integers(low=0, high=40)
                    desired_rms = np.sqrt(dominant_rms ** 2 / 10 ** (snr / 10.0))
            else:
                desired_rms = 0.06 * self.rng.random() + 0.07
            source_sound = self.normalize_audio(source_sound, desired_rms=desired_rms)
            source_sounds.append(source_sound)
        return source_sounds


    def obtain_binaural_rir_with_reverb(self, pair_path, camera_ind, source_ind):
        if self.args.indirect_ratio is None:
        # load binaural rir
            binaural_rir_path = os.path.join(pair_path, 'binaural_rirs', f'sound_{source_ind}_camera_{camera_ind}_rir.wav')
            binaural_rir, _ = sf.read(binaural_rir_path, dtype='float32', always_2d=True)
        else:
            direct_rir, _ = sf.read(os.path.join(pair_path, 'binaural_rirs_direct', f'sound_{source_ind}_camera_{camera_ind}_rir.wav'), dtype='float32', always_2d=True)
            indirect_rir, _  = sf.read(os.path.join(pair_path, 'binaural_rirs_indirect', f'sound_{source_ind}_camera_{camera_ind}_rir.wav'), dtype='float32', always_2d=True)
            zero_padding = np.zeros((indirect_rir.shape[0] - direct_rir.shape[0], direct_rir.shape[1]))
            direct_rir = np.concatenate((direct_rir, zero_padding), axis=0)
            binaural_rir = direct_rir + indirect_rir * self.args.indirect_ratio
        return binaural_rir


    def generate_audio(self, index, pair_path, camera_ind, source_sounds):
        # import pdb; pdb.set_trace()
        audio = None
        dominant_rms = None
        for source_ind in range(self.args.n_source):
            if self.online_render:
                # load binaural rir
                binaural_rir = self.obtain_binaural_rir_with_reverb(pair_path, camera_ind, source_ind)
                source_sound = source_sounds[source_ind]
                render_audio = self.impulse_response_to_sound(binaural_rir, source_sound)
            else:
                render_audio_path = os.path.join(pair_path, 'render_audios', f'sound_{source_ind}_camera_{camera_ind}_audio.wav')
                render_audio, _ = self.read_audio(render_audio_path, start=0, stop=np.rint(self.pr.clip_length * self.pr.samp_sr * 3).astype(int))
                render_audio = render_audio.T
            
            if self.args.with_dominant_sound and self.args.ssl_flag:
                if source_ind == 0:
                    dominant_rms = desired_rms = np.sqrt(np.mean(render_audio**2))
                else:
                    snr = self.args.dominant_snr if self.args.dominant_snr else self.rng.integers(low=5, high=30)
                    desired_rms = np.sqrt(dominant_rms ** 2 / 10 ** (snr / 10.0))
                render_audio = self.normalize_audio(render_audio, desired_rms=desired_rms)

            if audio is None:
                audio = render_audio
            else:
                audio += render_audio   # shape as (C, L)
        
        # may move this noise part to other place 
        if self.args.add_noise:
            audio = self.add_gaussian_noise_by_snr(audio)
        
        if self.args.save_audio:
            # import pdb; pdb.set_trace()
            save_folder = os.path.join('./checkpoints', self.args.exp, 'saved_audio')
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f'audio_{str(index).zfill(5)}_camera_{camera_ind}.wav')
            sf.write(save_path, audio.T, self.pr.samp_sr)

        return audio


    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.
        return samples 


    def sum2audio(self, audio_1, audio_2):
        audio = audio_1 + audio_2
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio


    def impulse_response_to_sound(self, binaural_rir, source_sound):
        '''
            goal: create sound based on simulate impulse response
            binaural_rir: (num_sample, num_channel)
            source_sound: mono sound, (num_sample)
            rir and source sound should have same sampling rate
        '''
        # import pdb; pdb.set_trace()
        audio_length = source_sound.shape[0]
        binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel]) for channel in range(binaural_rir.shape[-1])])
        binaural_convolved = binaural_convolved[:, :audio_length]
        return binaural_convolved


    def add_gaussian_noise_by_snr(self, signal, snr=None):
        # import pdb; pdb.set_trace()
        snr = snr if snr is not None else self.rng.integers(low=10, high=40)
        signal_rms = np.sqrt(np.mean(signal ** 2))
        if signal_rms == 0:
            return signal
        noise_rms = np.sqrt(signal_rms ** 2 / 10 ** (snr / 10.0))
        noise = self.rng.normal(loc=0.0, scale=noise_rms, size=signal.shape)
        audio = signal + noise
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio


    def read_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)
        return image
    
    
    def generate_image_transform(self, args, pr):
        resize_funct = transforms.Resize(pr.img_size)
        vision_transform_list = [
            resize_funct,
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return vision_transform_list


    def relative_angle_distribution(self):
        list_sample = self.get_list_sample(self.pr.list_train)
        angle_distribution = []
        for i in range(len(list_sample)):
            info = list_sample[i]
            pair_path = info['path']
            meta_path = os.path.join(pair_path, 'metadata.json')
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
            camera_angle_list = [meta_dict['camera_0_angle'], meta_dict['camera_1_angle']]
            relative_angle = self.calculate_relative_rotation(camera_angle_list)
            angle_distribution.append(relative_angle)
            angle_distribution.append(-relative_angle)
        return np.array(angle_distribution)


    def calculate_relative_rotation(self, camera_angle_list, return_bin=False):
        '''
            We define turning left as +, turning right as -.
            relative_angle are within (-180, 180]
        '''
        relative_angle = camera_angle_list[1] - camera_angle_list[0]
        if relative_angle >= 180: 
            relative_angle = relative_angle - 360
        elif relative_angle < -180:
            relative_angle = 360 + relative_angle
        if not return_bin:
            return relative_angle
        else:
            offset = 1e-2
            angle_range = np.abs(self.angle_distribution).max() - np.abs(self.angle_distribution).min()
            bin_size = angle_range / (self.pr.num_classes // 2)
            if relative_angle >= 0:
                relative_angle_bin = (relative_angle - np.abs(self.angle_distribution).min()) // bin_size 
                relative_angle_bin = np.clip(relative_angle_bin, 0, self.pr.num_classes // 2 - 1) + self.pr.num_classes // 2
            elif relative_angle < 0:
                relative_angle_bin = (relative_angle + np.abs(self.angle_distribution).max()) // bin_size
                relative_angle_bin = np.clip(relative_angle_bin, 0, self.pr.num_classes // 2 - 1)
            return relative_angle, relative_angle_bin
    

    def source_direction_distribution(self):
        # import pdb; pdb.set_trace()
        list_sample = self.get_list_sample(self.pr.list_train)
        direction_distribution = []
        for i in range(len(list_sample)):
            info = list_sample[i]
            pair_path = info['path']
            meta_path = os.path.join(pair_path, 'metadata.json')
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
            for key in meta_dict.keys():
                if key.find('relative_angle_between_sound') != -1:
                    direction_distribution.append(meta_dict[key])
        return np.array(direction_distribution)


    def calc_source_direction_to_bin(self, source_angle):
        '''
            We define turning left as +, turning right as -.
            source angle are within (-180, 180]
        '''
        angle_range = self.source_distribution.max() - self.source_distribution.min()
        bin_size = angle_range / self.pr.num_classes      
        source_angle_bin = (source_angle - self.source_distribution.min()) // bin_size 
        source_angle_bin = np.clip(source_angle_bin, 0, self.pr.num_classes - 1)
        return source_angle_bin



