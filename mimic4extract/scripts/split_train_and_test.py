from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
import random
import pandas as pd


def generate_test_val_id(all_stays):
    all_id = list(set(all_stays.subject_id))
    test_nums = int(len(all_id) * 0.2)
    val_nums = int(len(all_id) * 0.1)

    test_choose = [1] * test_nums + [0] * (len(all_id) - test_nums)
    random.seed(42)
    random.shuffle(test_choose)
    testset = pd.DataFrame([all_id, test_choose])
    testset.T.to_csv(os.path.join(os.path.dirname(__file__), '../resources/testset.csv'), index=False, header=False)

    val_choose = [1] * val_nums + [0] * (len(all_id) - test_nums - val_nums)
    random.shuffle(val_choose)
    valset = testset.T[testset.T[1] == 0]
    valset[1] = val_choose
    valset.to_csv(os.path.join(os.path.dirname(__file__), '../resources/valset.csv'), index=False, header=False)


def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)


def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    # generate_test_val_id(pd.read_csv(os.path.join(args.subjects_root_path, 'all_stays.csv')))

    test_set = set()

    with open(os.path.join(os.path.dirname(__file__), '../resources/testset.csv'), "r") as test_set_file:
        for line in test_set_file:
            x, y = line.split(',')
            if int(y) == 1:
                test_set.add(x)

    folders = os.listdir(args.subjects_root_path)
    folders = list((filter(str.isdigit, folders)))
    train_patients = [x for x in folders if x not in test_set]
    test_patients = [x for x in folders if x in test_set]
    
    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")


if __name__ == '__main__':
    main()

