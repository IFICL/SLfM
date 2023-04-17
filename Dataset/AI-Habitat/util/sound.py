from util import make_configuration, add_acoustic_config
from tqdm import tqdm
import argparse
import math
import os
import json
import shutil
import glob

import numpy as np
import random

import habitat
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

import soundfile as sf
from scipy.io import wavfile
from scipy.signal import fftconvolve

import sys
sys.path.append('..')


def normalize_audio(samples, desired_rms=0.1, eps=1e-4):
    # import pdb; pdb.set_trace()
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    samples[samples > 1.] = 1.
    samples[samples < -1.] = -1.
    return samples 


def impulse_response_to_sound(binaural_rir, source_sound, sampling_rate):
    '''
        goal: create sound based on simulate impulse response
        binaural_rir: (num_sample, num_channel)
        source_sound: mono sound, (num_sample)
        rir and source sound should have same sampling rate
    '''
    binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel]) for channel in range(binaural_rir.shape[-1])])
    return binaural_convolved


def sample_audio_database(args, settings):
    if args.audio_database == 'FMA':
        database = glob.glob('Free-Music-Archive/ProcessedData/*/*/*.wav')
    elif args.audio_database == 'LibriSpeech':
        database = glob.glob('LibriSpeech/ProcessedData/*/*/*/audio.wav')
    database.sort()
    sampled_audio_paths = np.random.choice(database, args.num_source, replace=False)
    source_sounds = []
    for audio_path in sampled_audio_paths:
        source_sound, _ = sf.read(audio_path, start=0, stop=int(settings['sample_rate'] * 10), dtype='float32', always_2d=True)
        source_sound = source_sound.mean(-1)
        desired_rms = 0.03 * np.random.rand() + 0.07
        source_sound = normalize_audio(source_sound, desired_rms=desired_rms)
        source_sounds.append(source_sound)
    return source_sounds


def calc_sound_direction_for_agent(pos_sound, pos_agent, angle_agent):
    '''
        Calculate sound direction respected to current agent's position and its orientation in 2D space
        where +Y is upward, -Z is forward and +X is rightward.
        rotation Left is + and right is - 
    '''
    pos_agent_2d = np.array([-pos_agent[2], -pos_agent[0]])
    pos_sound_2d = np.array([-pos_sound[2], -pos_sound[0]])
    vector_sound2agent = pos_sound_2d - pos_agent_2d
    if np.linalg.norm(vector_sound2agent) == 0:
        vector_sound2agent = np.array([1, 0])
    vector_sound2agent = vector_sound2agent / np.linalg.norm(vector_sound2agent)
    vector_agent = np.array([np.cos(angle_agent / 180 * np.pi), np.sin(angle_agent / 180 * np.pi)])
    dot_product = np.dot(vector_sound2agent, vector_agent)
    cross_product = np.cross(vector_agent, vector_sound2agent)
    angle = np.rad2deg(np.arccos(dot_product)) 
    if cross_product < 0:
        angle = - angle
    return angle

def verify_rir_volume(rir):
    rir_rms = np.sqrt(np.mean(rir ** 2))
    rms_th = 1e-6
    if rir_rms < rms_th:
        return False
    else:
        return True

def verify_unblocked_sound_source(sim, pos_source, pos_agent):
    # import pdb; pdb.set_trace()
    # euclidean distance
    euclidean_distance = np.sqrt(((pos_source - pos_agent) ** 2).sum())
    if euclidean_distance < 0.1:
        return False
    # geodesic distance
    path = habitat_sim.nav.ShortestPath()
    path.requested_start = pos_source
    path.requested_end = pos_agent
    found_path = sim.pathfinder.find_path(path)
    if not found_path:
        return False

    geodesic_distance = path.geodesic_distance
    status = np.isclose(geodesic_distance, euclidean_distance, atol=1e-3)
    return status


def sample_sound_source_location(args, sim, settings, receivers):
    '''
        Goal: to sample given number of sound source localization (x1, y1, z1) with receiver distance constraints
    '''
    # import pdb; pdb.set_trace()
    n_source = args.num_source
    sources = []
    count = 0

    center_pos = np.zeros(3)
    for i in range(len(receivers)):
        center_pos += receivers[i]['position']
    center_pos = center_pos / len(receivers)
    center_pos += np.array([0, settings["sensor_height"], 0])
    # import pdb; pdb.set_trace()
    maximum_try = 500 * n_source
    try_count = 0
    while True:
        if count >= n_source:
            break
        pos = sim.pathfinder.get_random_navigable_point_near(circle_center=center_pos, radius=3, max_tries=1000)
        pos = np.array(pos)
        status = 0
        for camera_id in range(len(receivers)):
            status += verify_unblocked_sound_source(sim, pos, receivers[i]['position'])
        
        if status < len(receivers):
            try_count += 1
            if try_count > maximum_try:
                break
            continue
        pos[1] = pos[1] + np.random.rand() * 1 + 0.7
        sources.append(np.array(pos))
        count += 1

    if len(sources) < n_source:
        return None
    return sources


def sample_sound_source_location_for_camera_set(args, scene_id, settings, camera_sets):
    '''
        Goal: to sample given number of sound source localization (x1, y1, z1) with receiver distance constraints for all camera_sets
    '''
    cfg = make_configuration(args, scene_id, settings, add_semantic=False, visual_sensor=False)
    sim = habitat_sim.Simulator(cfg)
    sim = add_acoustic_config(sim, args, settings, indirect=False)
    sim.seed(settings['seed'])

    # import pdb; pdb.set_trace()
    camera_sets_with_sound = []
    for set_ind, camera_set in tqdm(enumerate(camera_sets), total=len(camera_sets), desc='sampling sound source'):
        # import pdb; pdb.set_trace()
        sources = sample_sound_source_location(args, sim, settings, camera_set)
        if sources is None:
            continue

        break_flag = False
        for camera_ind, camera in enumerate(camera_set):
            agent = sim.get_agent(0)
            new_state = sim.get_agent(0).get_state()
            new_state.position = camera['position']
            new_state.rotation = quat_from_angle_axis(math.radians(camera['angle']), np.array([0, 1, 0]))
            new_state.sensor_states = {}
            agent.set_state(new_state, True)
            if args.rir_to_sound:
                source_sounds = sample_audio_database(args, settings)

            for source_ind, source in enumerate(sources):
                # Get the audio sensor object
                audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
                # set audio source location, no need to set the agent location, will be set implicitly
                audio_sensor.setAudioSourceTransform(np.array(source))

                observation = sim.get_sensor_observations()
                
                if audio_sensor.sourceIsVisible():
                    continue
                
                break_flag = True
                break # jump out of source

            if break_flag:
                break  # jump out of current camera_set
        
        if break_flag:
            continue

        # continue if the sources meet the requirement
        camera_set_info = {
            'cameras': camera_set,
            'sources': sources,
        }
        if args.rir_to_sound:
            camera_set_info['source_sounds'] = source_sounds
        
        camera_sets_with_sound.append(camera_set_info)

    sim.close()
    return camera_sets_with_sound