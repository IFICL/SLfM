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


def make_default_settings(args):
    settings = {
        "width": 320,  # Spatial resolution of the observations
        "height": 240,
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
        "fov": 60.0,
        "sample_rate": 16000,
        "channel_type": 'Binaural',
        'channel_count': 2
    }
    if args.high_resol:
        ratio = 3
        settings["width"] = settings["width"] * ratio
        settings["height"] = settings["height"] * ratio

    return settings


def make_configuration(args, scene_id, settings, add_semantic=False, visual_sensor=True):
    # import pdb; pdb.set_trace()
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    if args.dataset == 'hm3d':
        backend_cfg.scene_dataset_config_file = "Habitat/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    elif args.dataset == 'gibson':
        backend_cfg.scene_dataset_config_file = "Habitat/scene_datasets/gibson/gibson_semantic.scene_dataset_config.json"
    
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = settings["enable_physics"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    if visual_sensor: 
        # agent configuration
        # RGB sensor
        rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
        rgb_sensor_cfg.resolution = [settings["height"], settings["width"]]
        rgb_sensor_cfg.far = np.iinfo(np.int32).max
        rgb_sensor_cfg.hfov = mn.Deg(settings["fov"])
        rgb_sensor_cfg.position = [0, settings["sensor_height"], 0]
        rgb_sensor_cfg.orientation = [settings["sensor_pitch"], 0.0, 0.0]

        # Depth sensor
        depth_sensor_cfg = habitat_sim.CameraSensorSpec()
        depth_sensor_cfg.uuid = 'depth_camera'
        depth_sensor_cfg.resolution = [settings["height"], settings["width"]]
        depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_cfg.hfov = mn.Deg(settings["fov"])
        depth_sensor_cfg.position = [0, settings["sensor_height"], 0]
        depth_sensor_cfg.orientation = [settings["sensor_pitch"], 0.0, 0.0]

        agent_cfg.sensor_specifications = [rgb_sensor_cfg, depth_sensor_cfg]

        if add_semantic:
            # semantic sensor
            semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
            semantic_sensor_cfg.uuid = 'semantic_camera'
            semantic_sensor_cfg.resolution = [settings["height"], settings["width"]]
            semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_cfg.hfov = mn.Deg(settings["fov"])
            semantic_sensor_cfg.position = [0, settings["sensor_height"], 0]
            semantic_sensor_cfg.orientation = [settings["sensor_pitch"], 0.0, 0.0]
            agent_cfg.sensor_specifications.append(semantic_sensor_cfg)
    else:
        agent_cfg.sensor_specifications = []
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return cfg


def add_acoustic_config(sim, args, settings, direct=False, indirect=False):
    '''
        Detail document can be seen here: https://github.com/facebookresearch/habitat-sim/blob/main/docs/AUDIO.md#steps-to-run-audio-simulation-in-python
    '''
    # create the acoustic configs
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = False
    # audio_sensor_spec.enableMaterials = True
    audio_sensor_spec.channelLayout.type = getattr(habitat_sim.sensor.RLRAudioPropagationChannelLayoutType, settings["channel_type"])
    audio_sensor_spec.channelLayout.channelCount = settings["channel_count"]
    audio_sensor_spec.position = [0, settings["sensor_height"], 0]
    audio_sensor_spec.acousticsConfig.sampleRate = settings["sample_rate"]
    audio_sensor_spec.acousticsConfig.threadCount = 2
    audio_sensor_spec.acousticsConfig.direct = direct
    audio_sensor_spec.acousticsConfig.indirect = indirect

    # add the audio sensor
    sim.add_sensor(audio_sensor_spec)

    # add material property
    if args.dataset in ['mp3d', 'gibson', 'hm3d']:
        audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
        audio_sensor.setAudioMaterialsJSON('Habitat/scene_datasets/mp3d_material_config.json')
    
    return sim