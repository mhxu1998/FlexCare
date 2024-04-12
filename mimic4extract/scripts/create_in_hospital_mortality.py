from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
from mimic4extract.scripts.utils import merge_multimodal_data, create_train_val_test_set


def process_time_series_with_label(args, eps=1e-6, n_hours=48):
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(args.root_path)))
    for patient in tqdm(patients, desc='Iterating over patients'):
        patient_folder = os.path.join(args.root_path, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue
                icustay = label_df['Icustay'].iloc[0]
                
                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename

                # no time series file in ehr folder
                if os.path.isfile(os.path.join(args.ehr_path, output_ts_filename)) is not True:
                    continue

                xy_pairs.append((patient, icustay, output_ts_filename, mortality))
                # xy_pairs.append((output_ts_filename, icustay, mortality))

    print("Number of created samples:", len(xy_pairs))
    xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('subject_id,stay_id,time_series,period_length,y_true\n')
        for (patient, icustay, ts_filename, y) in xy_pairs:
            listfile.write('{},{},{},0,{:d}\n'.format(patient, icustay, ts_filename, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder.")
    parser.add_argument('ehr_path', type=str, help="Path to time series data folder.")
    parser.add_argument('cxr_path', type=str, help="Path to cxr data folder.")
    parser.add_argument('note_path', type=str, help="Path to note data folder.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_time_series_with_label(args)

    # merge multimodal data
    merge_multimodal_data(args, 'in-hospital-mortality')

    # split into tr/val/te set
    create_train_val_test_set(args)


if __name__ == '__main__':
    main()
