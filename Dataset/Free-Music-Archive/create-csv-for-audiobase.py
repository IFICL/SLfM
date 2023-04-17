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

parser = argparse.ArgumentParser(description="""Configure""")
parser.add_argument('--type', default='', type=str)
parser.add_argument('--data_split', default='8:1:1', type=str)
parser.add_argument('--unshuffle', default=False, action='store_true')
random.seed(1234)


def write_csv(data_list, filepath):
    # import pdb; pdb.set_trace()
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, delimiter=',',
                                fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for info in data_list:
            writer.writerow(info)
    print('{} items saved to {}.'.format(len(data_list), filepath))


def create_list_for_video(args, name, video_list):
    # import pdb; pdb.set_trace()
    sample_list = []

    for video in tqdm(video_list):
        video_path = os.path.join(*video.split('/')[:-1])
        path = os.path.join('./data/AI-Habitat/Free-Music-Archive', video_path)
        sample_list.append({'path': path})
    if not args.unshuffle:
        random.shuffle(sample_list)
    return sample_list


def main(args):
    # import pdb; pdb.set_trace()
    read_path = './ProcessedData'
    split_path = f'./data-split'
    if args.type != '':
        split_path = os.path.join(split_path, args.type)
    os.makedirs(split_path, exist_ok=True)

    data_list = glob.glob(f'{read_path}/*/*/*.wav')
    data_list.sort()
    if not args.unshuffle:
        random.shuffle(data_list)

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

    csv_zip = zip(['train', 'val', 'test'], [
                  train_list, valid_list, test_list])
    for name, video_list in tqdm(csv_zip):
        if len(video_list) == 0:
            continue
        sample_list = create_list_for_video(args, name, video_list)
        csv_name = f'{split_path}/{name}.csv'
        write_csv(sample_list, csv_name)


# Usage: python create-csv-for-audiobase.py --type='FMA' --data_split='8:1:1'

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
