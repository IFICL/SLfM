import itertools
from tqdm import tqdm
import argparse
import math
import os
import json
import shutil
import glob
import itertools

import magnum as mn
import numpy as np
import random

import habitat
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

import sys
sys.path.append('..')
from util import make_configuration


def camera_pair_matching(args, cameras, i, j):
    # check agent in the sample plane or not
    if cameras[i]['position'][1] != cameras[j]['position'][1]:
        return False
    
    camera_distance = np.sqrt(np.sum((cameras[i]['position'] - cameras[j]['position']) ** 2))
    angle_difference = np.abs(cameras[i]["angle"] - cameras[j]["angle"])
    if args.small_motion:
        if camera_distance > 2:
            return False

    sem1 = cameras[i]['semantic_raw']
    sem2 = cameras[j]['semantic_raw']
    max_ins = max(sem1.max(), sem2.max())
    cam1_instances = np.bincount(sem1.flatten(), minlength=max_ins)
    cam1_instances[cam1_instances < 50] = 0
    cam1_ids = np.nonzero(cam1_instances)[0]
    cam2_instances = np.bincount(sem2.flatten(), minlength=max_ins)
    cam2_instances[cam2_instances < 50] = 0
    cam2_ids = np.nonzero(cam2_instances)[0]
    inte = np.intersect1d(cam1_ids, cam2_ids)
    if inte.size < 15:
        return False
    return True


def sample_camera_set_with_n_view(args, scene_id, settings):
    '''
        Goal: to sample given number of camera pairs [(x1, y1, z1, angle1), (x2, y2, z2, angle2)] with overlap constraints
    '''
    if args.dataset == 'replica': 
        semantic_scene_id = scene_id.replace('mesh.ply', 'habitat/mesh_semantic.ply')
    elif args.dataset == 'hm3d': 
        semantic_scene_id = scene_id
    cfg = make_configuration(args, semantic_scene_id, settings, add_semantic=True)
    sim = habitat_sim.Simulator(cfg)
    sim.seed(settings['seed'])

    # import pdb; pdb.set_trace()
    n_set = args.num_per_scene
    cameras = []
    n_cameras = args.num_camera
    n_view = args.num_view

    for i in range(n_cameras):
        # find a random position and rotation
        pos = sim.pathfinder.get_random_navigable_point()
        angle = random.random() * math.pi * 2
        # render an image
        agent = sim.get_agent(0)
        new_state = sim.get_agent(0).get_state()
        new_state.position = pos
        new_state.rotation = quat_from_angle_axis(angle, np.array([0, 1, 0]))
        new_state.sensor_states = {}
        agent.set_state(new_state, True)
        observation = sim.get_sensor_observations()
        rgb = observation["rgba_camera"][..., :3]
        depth = observation['depth_camera']
        # this is actually instance segmentation
        semantic_raw = observation['semantic_camera']
        # filter unqualified results
        depth[np.isnan(depth)] = 0.0
        depth_max = depth.max()
        depth_mean = depth.mean()

        if depth_max < 3.0 or depth_mean < 1.0:
            continue
        camera_entry = {
            'position': np.array(pos),
            'angle': angle / (math.pi * 2) * 360,
            'rgb': rgb,
            'depth': depth,
            'semantic_raw': semantic_raw,
        }
        cameras.append(camera_entry)

    camera_set_dict = {}
    for i in range(len(cameras)):
        camera_set_dict[i] = []
        for j in range(len(cameras)):
            if i == j:
                continue
            
            if camera_pair_matching(args, cameras, i, j):
                camera_set_dict[i].append(j)
    
    # create camera set from camera set dictinatory
    camera_sets = []
    for i in range(len(cameras)):
        if len(camera_set_dict[i]) < n_view - 1:
            continue
        for c in itertools.combinations(camera_set_dict[i], n_view - 1):
            camera_set = []
            camera_set.append({key: cameras[i][key] for key in ['position', 'angle']})
            for ind in c:
                camera_set.append({key: cameras[ind][key] for key in ['position', 'angle']})
            camera_sets.append(camera_set)
    
    if len(camera_sets) > n_set:
        camera_sets = random.sample(camera_sets, n_set)
    sim.close()
    return camera_sets




def camera_pair_matching_for_rotation(args, settings, cameras, i, j):
    # check if the camera is the same or not
    if i == j:
        return False
    
    # check if angle matches the require
    angle_difference = np.abs(cameras[i]["angle"] - cameras[j]["angle"])
    angle_difference = min(angle_difference, 360 - angle_difference)
    if angle_difference >= (settings['fov'] * 1.5) or angle_difference < 10:
        return False
    
    sem1 = cameras[i]['semantic_raw']
    sem2 = cameras[j]['semantic_raw']
    max_ins = max(sem1.max(), sem2.max())
    cam1_instances = np.bincount(sem1.flatten(), minlength=max_ins)
    cam1_instances[cam1_instances < 50] = 0
    cam1_ids = np.nonzero(cam1_instances)[0]
    cam2_instances = np.bincount(sem2.flatten(), minlength=max_ins)
    cam2_instances[cam2_instances < 50] = 0
    cam2_ids = np.nonzero(cam2_instances)[0]
    inte = np.intersect1d(cam1_ids, cam2_ids)
    if inte.size < 10:
        return False
    return True


def sample_rotated_camera_set_with_n_view(args, scene_id, settings):
    '''
        Goal: to sample given number of camera pairs [(x1, y1, z1, angle1), (x2, y2, z2, angle2), ...] with overlap constraints and only rotation
    '''
    # import pdb; pdb.set_trace()
    if args.dataset == 'replica': 
        semantic_scene_id = scene_id.replace('mesh.ply', 'habitat/mesh_semantic.ply')
    elif args.dataset in ['hm3d', 'gibson']: 
        semantic_scene_id = scene_id
    cfg = make_configuration(args, semantic_scene_id, settings, add_semantic=True)
    sim = habitat_sim.Simulator(cfg)
    sim.seed(settings['seed'])

    n_set = args.num_per_scene
    n_pos = args.num_camera # better to be 100 or 200
    n_angle = args.num_angle
    n_set_per_pos = int(n_set // n_pos) if n_set >= n_pos else 1
    n_view = args.num_view
    final_camera_sets = []
    
    for i in range(n_pos):
        cameras = []
        # find a random position
        center_pos = sim.pathfinder.get_random_navigable_point(max_tries=500)
        if not sim.pathfinder.is_navigable(center_pos):
            continue
        # find random angles
        angles = np.random.random(n_angle) * math.pi * 2
        for angle in angles:
            # angle = random.random() * math.pi * 2
            # import pdb; pdb.set_trace()
            if args.add_small_translation:
                pos = sim.pathfinder.get_random_navigable_point_near(circle_center=center_pos, radius=0.25, max_tries=1000)
            else:
                pos = center_pos
            
            if not sim.pathfinder.is_navigable(pos):
                pos = center_pos

            # render an image
            agent = sim.get_agent(0)
            new_state = sim.get_agent(0).get_state()
            new_state.position = pos
            new_state.rotation = quat_from_angle_axis(angle, np.array([0, 1, 0]))
            new_state.sensor_states = {}
            agent.set_state(new_state, True)
            observation = sim.get_sensor_observations()
            rgb = observation["rgba_camera"][..., :3]
            depth = observation['depth_camera']
            # this is actually instance segmentation
            semantic_raw = observation['semantic_camera']
            # filter unqualified results
            depth[np.isnan(depth)] = 0.0
            depth_max = depth.max()
            depth_mean = depth.mean()

            if not args.random_angle:
                if (depth_max < 4.0 or depth_mean < 1.0):
                    continue
                

            camera_entry = {
                'position': np.array(pos),
                'angle': angle / (math.pi * 2) * 360,
                'rgb': rgb,
                'depth': depth,
                'semantic_raw': semantic_raw,
            }
            cameras.append(camera_entry)

        camera_sets = []
        # sample camera sets with same position but different angles
        if args.random_angle:
            # This sampling process randomly sampling n views with angles
            if len(cameras) < max(int(n_angle / 2), n_view):  # remove this position if it doesn't have enough number
                continue
            camera_set = []
            for ind in range(len(cameras)):
                camera_set.append({key: cameras[ind][key] for key in ['position', 'angle']})
            camera_set = random.sample(camera_set, n_view)
            camera_sets.append(camera_set)
        else:
            # This sampling process ensure other views are correlated to the first view
            camera_set_dict = {}
            for ind in range(len(cameras)):
                camera_set_dict[ind] = []
                for jnd in range(len(cameras)):
                    if camera_pair_matching_for_rotation(args, settings, cameras, ind, jnd):
                        camera_set_dict[ind].append(jnd)

            # create camera set from camera set dictinatory
            for ind in range(len(cameras)):
                if len(camera_set_dict[ind]) < n_view - 1:
                    continue
                for c in itertools.combinations(camera_set_dict[ind], n_view - 1):
                    camera_set = []
                    camera_set.append({key: cameras[ind][key] for key in ['position', 'angle']})
                    for jnd in c:
                        camera_set.append({key: cameras[jnd][key] for key in ['position', 'angle']})
                    camera_sets.append(camera_set)
            if len(camera_sets) > n_set_per_pos:
                camera_sets = random.sample(camera_sets, n_set_per_pos)
        # import pdb; pdb.set_trace()
        final_camera_sets += camera_sets

    sim.close()
    return final_camera_sets