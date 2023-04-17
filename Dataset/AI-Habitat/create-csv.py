import os
import numpy as np
import glob
import argparse
import random
import json
from tqdm import tqdm
import csv
import cv2
import soundfile as sf
import shutil

parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--max_sample', default=-1, type=int)
parser.add_argument('--dataset', default='', type=str)
parser.add_argument('--type', default='', type=str)
parser.add_argument('--data_split', default='8:1:1', type=str)
parser.add_argument('--unshuffle', default=False, action='store_true')
parser.add_argument('--filter_angle', default=False, action='store_true')
parser.add_argument('--remove_invalid', default=False, action='store_true')

random.seed(1234)


def write_csv(data_list, filepath):
    # import pdb; pdb.set_trace()
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            writer.writerow(info)
    print('{} items saved to {}.'.format(len(data_list), filepath))


def create_list_for_scene(args, name, scene_list):
    # import pdb; pdb.set_trace()
    sample_list = []

    for scene in tqdm(scene_list):
        pair_list = glob.glob(f'{scene}/*')
        pair_list.sort()
        for pair in pair_list:
            # import pdb; pdb.set_trace()
            meta_path = os.path.join(pair, 'metadata.json')
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
            sound_angles = []
            for key in meta_dict.keys():
                if key.find('relative_angle_between_sound') != -1:
                    sound_angles.append(meta_dict[key])

            direct_rirs = glob.glob(f"{pair}/binaural_rirs_direct/*")
            indirect_rirs = glob.glob(f"{pair}/binaural_rirs_indirect/*")

            if len(direct_rirs) != len(indirect_rirs):
                if args.remove_invalid:
                    shutil.rmtree(pair)
                continue

            flag = np.isnan(sound_angles).sum()
            if flag > 0:
                if args.remove_invalid:
                    shutil.rmtree(pair)
                continue
            
            if args.filter_angle and name in ['train', 'val']:
                angle_flag = (np.array(sound_angles) > 90).sum() + (np.array(sound_angles) < -90).sum()
                if angle_flag > 0:
                    continue
            
            if args.filter_angle and name in ['test']:
                first_angle = float(meta_dict["relative_angle_between_sound_0_camera_0"])
                angle_flag = (first_angle > 90) or (first_angle < -90)
                if angle_flag:
                    continue

            path = os.path.join('./data/AI-Habitat', pair)
            sample = {
                'path': path
            }
            sample_list.append(sample)
    if not args.unshuffle:
        random.shuffle(sample_list)
    return sample_list


def main(args):
    # import pdb; pdb.set_trace()
    read_path = f'ProcessedData/{args.dataset}'
    split_path = f'./data-split'
    if args.type != '':
        split_path = os.path.join(split_path, args.type)

    os.makedirs(split_path, exist_ok=True)

    data_list = glob.glob(f'{read_path}/*')
    data_list.sort()
    if not args.unshuffle:
        random.shuffle(data_list)
    train_list = []
    valid_list = []
    test_list = []

    begin = 0
    ratios = args.data_split.split(':')
    ratios = np.array(list(map(int, ratios)))
    ratios = ratios / ratios.sum()
    n_train = begin + ratios[0]
    n_val = n_train + ratios[1]
    n_test = n_val + ratios[2]
    num = len(data_list)
    train_list = data_list[int(num * begin): int(num * n_train)]
    valid_list = data_list[int(num * n_train): int(num * n_val)]
    test_list = data_list[int(num * n_val): int(num * n_test)]

    csv_zip = zip(['train', 'val', 'test'], [train_list, valid_list, test_list])
    for name, scene_list in tqdm(csv_zip):
        if len(scene_list) == 0:
            continue

        sample_list = create_list_for_scene(args, name, scene_list)
        csv_name = f'{split_path}/{name}.csv'
        write_csv(sample_list, csv_name)



# Usage: python create-csv.py --dataset='hm3d-4view-rotation' --type='hm3d-4view-rotation' --data_split='9:1:1'  --remove_invalid
# Usage: python create-csv.py --dataset='hm3d-8view-rotation-rotnce' --type='hm3d-8view-rotation-rotnce' --data_split='9:1:1'  --remove_invalid
# Usage: python create-csv.py --dataset='hm3d-3view-smallmotion' --type='hm3d-3view-smallmotion' --data_split='9:1:1'  --remove_invalid
# Usage: python create-csv.py --dataset='hm3d-3view-smalltrans' --type='hm3d-3view-smalltrans' --data_split='9:1:1'  --remove_invalid
# Usage: python create-csv.py --dataset='hm3d-3view-smalltrans' --type='hm3d-3view-smalltrans-filterangle' --data_split='9:1:1'  --filter_angle




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
