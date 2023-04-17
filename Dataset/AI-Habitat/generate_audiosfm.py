# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import argparse
import math
import os
import json
import shutil
import glob

import magnum as mn
import numpy as np
import random

import habitat
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

import soundfile as sf
from scipy.io import wavfile
from scipy.signal import fftconvolve

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 600

from pyroomacoustics.experimental.rt60 import measure_rt60

from util import *


def normalize_depth(depth):
    min_depth = 0
    max_depth = 10
    depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth


def get_visual_observation(args, scene_id, settings, camera_sets):
    scene = scene_id.split('/')[-2]
    scene_obs_dir = f'{args.output_dir}/{scene}'
    cfg = make_configuration(args, scene_id, settings)
    sim = habitat_sim.Simulator(cfg)
    sim.seed(settings['seed'])
    
    for pair_ind, camera_set in tqdm(enumerate(camera_sets), total=len(camera_sets), desc=scene):
        # import pdb; pdb.set_trace()
        scene_pair_dir = f'{scene_obs_dir}/pair-{str(pair_ind).zfill(4)}'
        os.makedirs(scene_pair_dir, exist_ok=True)

        meta_path = os.path.join(scene_pair_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
        else:
            meta_dict = {}
            meta_dict['scene_id'] = scene_id
        
        for camera_ind, camera in enumerate(camera_set['cameras']):
            agent = sim.get_agent(0)
            new_state = sim.get_agent(0).get_state()
            new_state.position = camera['position']
            new_state.rotation = quat_from_angle_axis(math.radians(camera['angle']), np.array([0, 1, 0]))
            new_state.sensor_states = {}
            agent.set_state(new_state, True)

            meta_dict[f'camera_{camera_ind}_position'] = (camera['position'] + np.array([0, settings["sensor_height"], 0])).tolist()
            meta_dict[f'camera_{camera_ind}_angle'] = camera['angle'] 
            # import pdb; pdb.set_trace()
            camera_rotation = quat_from_angle_axis(math.radians(camera['angle']), np.array([0, 1, 0])) * quat_from_angle_axis(-settings['sensor_pitch'], np.array([-1, 0, 0]))
            meta_dict[f'camera_{camera_ind}_rotation_in_quaternion'] = [camera_rotation.w, camera_rotation.x, camera_rotation.y, camera_rotation.z]
            
            observation = sim.get_sensor_observations()
            rgb = observation["rgba_camera"][..., :3]
            depth = normalize_depth(observation['depth_camera'])

            if args.high_resol:
                high_resol_dir = os.path.join(scene_pair_dir, 'high_resol')
                os.makedirs(high_resol_dir, exist_ok=True)
                plt.imsave(os.path.join(high_resol_dir, f'camera_{camera_ind}_rgb.png'), rgb)
                plt.imsave(os.path.join(high_resol_dir, f'camera_{camera_ind}_depth.png'), depth)
            else:
                plt.imsave(os.path.join(scene_pair_dir, f'camera_{camera_ind}_rgb.png'), rgb)
                plt.imsave(os.path.join(scene_pair_dir, f'camera_{camera_ind}_depth.png'), depth)

            with open(meta_path, 'w') as fo:
                json.dump(meta_dict, fo, indent=4, sort_keys=True)
    sim.close()


def get_audio_observation(args, scene_id, settings, camera_sets, direct=False, indirect=False):
    scene = scene_id.split('/')[-2]
    scene_obs_dir = f'{args.output_dir}/{scene}'
    cfg = make_configuration(args, scene_id, settings, visual_sensor=False)
    sim = habitat_sim.Simulator(cfg)
    sim = add_acoustic_config(sim, args, settings, direct, indirect)
    sim.seed(settings['seed'])
    if direct and not indirect:
        indriect_name = '_direct'
    elif not direct and indirect:
        indriect_name = '_indirect'
    else:
        indriect_name = ''

    for pair_ind, camera_set in tqdm(enumerate(camera_sets), total=len(camera_sets), desc=scene):
        # import pdb; pdb.set_trace()
        scene_pair_dir = f'{scene_obs_dir}/pair-{str(pair_ind).zfill(4)}'
        os.makedirs(scene_pair_dir, exist_ok=True)
        binaural_rir_dir = os.path.join(scene_pair_dir, f'binaural_rirs{indriect_name}')
        os.makedirs(binaural_rir_dir, exist_ok=True)

        if args.rir_to_sound:
            render_audio_dir = os.path.join(scene_pair_dir, f'render_audios{indriect_name}')
            os.makedirs(render_audio_dir, exist_ok=True)
            source_sounds = camera_set['source_sounds']
        
        meta_path = os.path.join(scene_pair_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
        else:
            meta_dict = {}
            meta_dict['scene_id'] = scene_id
        
        for camera_ind, camera in enumerate(camera_set['cameras']):
            agent = sim.get_agent(0)
            new_state = sim.get_agent(0).get_state()
            new_state.position = camera['position']

            new_state.rotation = quat_from_angle_axis(math.radians(camera['angle']), np.array([0, 1, 0]))
            new_state.sensor_states = {}
            agent.set_state(new_state, True)

            for source_ind, source in enumerate(camera_set['sources']):
                # Get the audio sensor object
                audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
                # set audio source location, no need to set the agent location, will be set implicitly
                audio_sensor.setAudioSourceTransform(np.array(source))

                observation = sim.get_sensor_observations()
                rir = np.array(observation['audio_sensor'])
                
                rir_save_path = os.path.join(binaural_rir_dir, f'sound_{source_ind}_camera_{camera_ind}_rir.wav')
                sf.write(rir_save_path, rir.T, settings["sample_rate"])

                if args.rir_to_sound:
                    render_audio = impulse_response_to_sound(rir.T, source_sounds[source_ind], settings["sample_rate"])
                    sf.write(os.path.join(render_audio_dir, f'sound_{source_ind}_camera_{camera_ind}_audio.wav'), render_audio.T, settings["sample_rate"])

                meta_dict[f'source_{source_ind}_position'] = source.tolist()
                sound_angle = calc_sound_direction_for_agent(source, camera['position'], camera['angle'])
                meta_dict[f'relative_angle_between_sound_{source_ind}_camera_{camera_ind}'] = sound_angle
                
                if indirect and direct: 
                    rt60_l = measure_rt60(rir[0], settings["sample_rate"], decay_db=30, plot=False)
                    rt60_r = measure_rt60(rir[1], settings["sample_rate"], decay_db=30, plot=False)
                    meta_dict[f'sound_{source_ind}_camera_{camera_ind}_rir_rt60'] = [rt60_l, rt60_r]
                    meta_dict[f'sound_{source_ind}_camera_{camera_ind}_RayEfficiency'] = audio_sensor.getRayEfficiency()

        with open(meta_path, 'w') as fo:
            json.dump(meta_dict, fo, indent=4, sort_keys=True)
    sim.close()


def run(args, scene_id, settings):
    # import pdb; pdb.set_trace()
    # where +Y is upward, -Z is forward and +X is rightward.
    np.random.seed(settings['seed'])
    random.seed(settings['seed'])

    scene = scene_id.split('/')[-2]
    scene_obs_dir = f'{args.output_dir}/{scene}'
    if args.dataset == 'hm3d':
        scene_id = scene_id.replace('semantic', 'basis')
    
    if args.rotation_only:
        camera_sets = sample_rotated_camera_set_with_n_view(args, scene_id, settings)
    else:
        camera_sets = sample_camera_set_with_n_view(args, scene_id, settings)

    camera_sets = sample_sound_source_location_for_camera_set(args, scene_id, settings, camera_sets)
    # import pdb; pdb.set_trace()
    get_visual_observation(args, scene_id, settings, camera_sets)
    get_audio_observation(args, scene_id, settings, camera_sets, direct=True, indirect=False)
    get_audio_observation(args, scene_id, settings, camera_sets, direct=False, indirect=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hm3d')
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--num-per-scene', type=int, default=1000, help='number of pair per scene')
    parser.add_argument('--num-camera', type=int, default=300,  help='number of sampled camera per scene')
    parser.add_argument('--num-angle', type=int, default=30)
    parser.add_argument('--num-view', type=int, default=2)
    parser.add_argument('--num-source', type=int, default=2)
    parser.add_argument('--reset', default=False, action='store_true')
    parser.add_argument('--rotation-only', default=False, action='store_true')
    parser.add_argument('--rir-to-sound', default=False, action='store_true')
    parser.add_argument('--audio_database',  type=str, default='LibriSpeech', choices=['FMA', 'LibriSpeech'])
    parser.add_argument('--small-motion', default=False, action='store_true')
    parser.add_argument('--add-small-translation', default=False, action='store_true')
    parser.add_argument('--random-angle', default=False, action='store_true')
    parser.add_argument('--high-resol', default=False, action='store_true')
    parser.add_argument('--split', type=int, default=0, help='i split of data to process')
    parser.add_argument('--total', type=int, default=1, help='total splits')

    args = parser.parse_args()
    if args.output_dir == '':
        args.output_dir = os.path.join('ProcessedData', args.dataset)
    else: 
        args.output_dir = os.path.join('ProcessedData', args.output_dir)
    
    if args.reset:
        shutil.rmtree(args.output_dir)

    if args.dataset == 'replica':
        scene_ids = glob.glob(f"Habitat/scene_datasets/replica/*/mesh.ply")
    elif args.dataset == 'gibson':
        scene_ids = glob.glob(f"Habitat/scene_datasets/gibson/*.glb")
    elif args.dataset == 'hm3d':
        scene_ids = glob.glob(f"Habitat/scene_datasets/hm3d/*/*/*.semantic.glb")
    else:
        raise ValueError

    scene_ids.sort()
    scene_ids = scene_ids[int(args.split / args.total * len(scene_ids)): int((args.split+1) / args.total * len(scene_ids))]
    settings = make_default_settings(args)
    
    for scene_id in tqdm(scene_ids, desc=f'Processing ID = {str(args.split).zfill(2)}'):
        run(args, scene_id, settings)



# Usage: python generate_audiosfm.py --dataset='hm3d' --output-dir='human-test' --rir-to-sound --num-per-scene=5 --num-source=2 --num-camera=10 --num-view=3 --num-angle=30 --rotation-only --split=0 --total=25
# Usage: python generate_audiosfm.py --dataset='hm3d' --output-dir='human-test' --rir-to-sound --num-per-scene=10 --num-source=1 --num-camera=5 --num-view=3 --num-angle=20 --rotation-only --split=0 --total=100 --high-resol

if __name__ == '__main__':
    main()