from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
import numpy as np
from mimic4extract.scripts.utils import merge_multimodal_data, create_train_val_test_set


def process_drg_with_multimodal(args, eps=1e-6):
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    note_file = os.path.join(args.note_path, 'note_all.csv')
    drg_label_file = os.path.join(args.mimic4_path, 'hosp/drgcodes.csv')
    all_stayfile = os.path.join(args.root_path, 'all_stays.csv')
    cxr_metafile = os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-metadata.csv')
    admissions_file = os.path.join(args.mimic4_path, 'hosp/admissions.csv')

    note = pd.read_csv(note_file)
    drg_label = pd.read_csv(drg_label_file)
    all_stay = pd.read_csv(all_stayfile)
    cxr_metadata = pd.read_csv(cxr_metafile)
    admissions = pd.read_csv(admissions_file)

    # Choose brief hospital course in the note
    note_data = note[['subject_id', 'hadm_id', 'brief_hospital_course']].dropna(subset=['brief_hospital_course'])

    # Choose the HCFA drg type
    drg_hcfa = drg_label[drg_label['drg_type']=='HCFA']

    # Merge drg label
    note_label = note_data.merge(drg_hcfa[['subject_id', 'hadm_id', 'drg_code']], on=['subject_id', 'hadm_id'], how='inner')

    # Remap drg code label
    unique_labels = sorted(set(note_label.drg_code))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = [label_mapping[label] for label in list(note_label.drg_code)]
    note_label['drg_code'] = remapped_labels

    # Merge corresponding ehr information
    note_ehr_data = note_label[['subject_id', 'hadm_id']].merge(all_stay[['subject_id', 'hadm_id', 'stay_id']], how='inner', on=['subject_id', 'hadm_id'])

    # stay_id->ts file/los
    st_map = dict()
    sp_map = dict()
    patients = list(set(note_ehr_data['subject_id'].astype(str).unique())&set(filter(str.isdigit, os.listdir(args.root_path))))
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

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                if os.path.isfile(os.path.join(args.ehr_path, output_ts_filename)) is not True:
                    continue

                st_map[icustay] = output_ts_filename
                sp_map[icustay] = los

    note_ehr_data.loc[:, 'time_series'] = note_ehr_data['stay_id'].map(st_map)
    note_ehr_data.loc[:, 'period_len'] = note_ehr_data['stay_id'].map(sp_map)

    # Merge corresponding cxr information
    cxr_data = cxr_metadata[cxr_metadata['ViewPosition'] == 'AP']
    cxr_data.loc[:, 'StudyTime'] = cxr_data['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
    cxr_data.loc[:, 'StudyDateTime'] = pd.to_datetime(
        cxr_data['StudyDate'].astype(str) + ' ' + cxr_data['StudyTime'].astype(str), format="%Y%m%d %H%M%S")
    cxr_hadm_data = cxr_data[['subject_id', 'study_id', 'dicom_id', 'StudyDateTime']].merge(admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']], how='inner', on='subject_id')
    cxr_hadm_data = cxr_hadm_data.loc[(cxr_hadm_data.StudyDateTime >= cxr_hadm_data.admittime) & ((cxr_hadm_data.StudyDateTime <= cxr_hadm_data.dischtime))]
    cxr_hadm_data_sorted = cxr_hadm_data.sort_values(by=['subject_id', 'StudyDateTime'], ascending=[True, True])
    cxr_hadm_data_final = cxr_hadm_data_sorted.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='last')

    note_cxr_data = note_label[['subject_id', 'hadm_id']].merge(cxr_hadm_data_final[['subject_id', 'hadm_id', 'dicom_id']], how='inner', on=['subject_id', 'hadm_id'])

    all_data = note_label[['subject_id', 'hadm_id', 'brief_hospital_course', 'drg_code']].merge(note_ehr_data[['subject_id', 'hadm_id', 'time_series', 'period_len']], how='left', on=['subject_id', 'hadm_id'])
    all_data = all_data.merge(note_cxr_data[['subject_id', 'hadm_id', 'dicom_id']], how='left', on=['subject_id', 'hadm_id'])
    all_data = all_data[['subject_id', 'hadm_id', 'time_series', 'period_len', 'dicom_id', 'brief_hospital_course', 'drg_code']]
    all_data.to_csv(os.path.join(output_dir, "multimodal_listfile.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('mimic4_path', type=str, help="Path to mimic4 folder.")
    parser.add_argument('root_path', type=str, help="Path to root folder.")
    parser.add_argument('ehr_path', type=str, help="Path to time series data folder.")
    parser.add_argument('cxr_path', type=str, help="Path to cxr data folder.")
    parser.add_argument('note_path', type=str, help="Path to note data folder.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_drg_with_multimodal(args)

    # split into tr/val/te set
    create_train_val_test_set(args)


if __name__ == '__main__':
    main()
