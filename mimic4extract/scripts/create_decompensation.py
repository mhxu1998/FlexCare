from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
random.seed(49297)
from tqdm import tqdm
from mimic4extract.scripts.utils import merge_multimodal_data, create_train_val_test_set, random_sample


def process_time_series_with_label(args, sample_rate=1.0, shortest_length=4.0, eps=1e-6, future_time_interval=24.0):

    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(args.root_path)))
    for patient in tqdm(patients, desc='Iterating over patients'):
        patient_folder = os.path.join(args.root_path, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("(length of stay is missing)", patient, ts_filename)
                    continue
                
                # import pdb; pdb.set_trace()
                
                stay = stays_df[stays_df.stay_id == label_df.iloc[0]['Icustay']]
                
                icustay = label_df['Icustay'].iloc[0]

                deathtime = stay['deathtime'].iloc[0]
                intime = stay['intime'].iloc[0]
                if pd.isnull(deathtime):
                    lived_time = 1e18
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("(no events in ICU) ", patient, ts_filename)
                    continue

                sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)

                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                output_ts_filename = patient + "_" + ts_filename

                for t in sample_times:
                    if mortality == 0:
                        cur_mortality = 0
                    else:
                        cur_mortality = int(lived_time - t < future_time_interval)
                    # xty_triples.append((output_ts_filename, t, icustay, cur_mortality))
                    xty_triples.append((patient, icustay, output_ts_filename, t, cur_mortality))

    print("Number of created samples:", len(xty_triples))
    xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('subject_id,stay_id,time_series,period_length,y_true\n')
        for (patient, icustay, ts_filename, t, y) in xty_triples:
            listfile.write('{},{},{},{:.6f},{:d}\n'.format(patient, icustay, ts_filename, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder.")
    parser.add_argument('ehr_path', type=str, help="Path to time series data folder.")
    parser.add_argument('cxr_path', type=str, help="Path to cxr data folder.")
    parser.add_argument('note_path', type=str, help="Path to note data folder.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_time_series_with_label(args)
    random_sample(args)

    # merge multimodal data
    merge_multimodal_data(args, 'decompensation')

    # split into tr/val/te set
    create_train_val_test_set(args)


if __name__ == '__main__':
    main()
