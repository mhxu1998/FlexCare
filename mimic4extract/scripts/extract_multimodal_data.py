from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(42)
from tqdm import tqdm
from mimic4extract.scripts.utils import split_train_val_test_id,process_note


def extract_time_series(args, eps=1e-6):
    output_dir = os.path.join(args.ehr_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    patients = list(filter(str.isdigit, os.listdir(args.root_path)))
    for patient in tqdm(patients):
        patient_folder = os.path.join(args.root_path, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if t > -eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)


def main():
    parser = argparse.ArgumentParser(description="Extract multimodal data for patients.")
    parser.add_argument('mimic4_path', type=str, help='Directory containing MIMIC-IV CSV files.')
    parser.add_argument('mimic_note_path', type=str, help='Directory containing MIMIC-NOTE CSV files.')
    parser.add_argument('root_path', type=str, help="Path to root folder patient information.")
    parser.add_argument('ehr_path', type=str, help="Directory where the time series data should be stored.")
    parser.add_argument('note_path', type=str, help="Directory where the note data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.ehr_path):
        os.makedirs(args.ehr_path)

    # Create the folder and move time series files into it, e.g., data/ehr
    print('create directory to store time series data')
    extract_time_series(args)

    # Extract required data from raw MIMIC-NOTE
    print('extract required data from raw MIMIC-NOTE')
    process_note(args.mimic_note_path, args.note_path)

    # split all patients into train/val/test sets
    print('split all patients into train/val/test sets')
    admissions = pd.read_csv(f'{args.mimic4_path}/hosp/admissions.csv')
    split_train_val_test_id(admissions)

if __name__ == '__main__':
    main()
