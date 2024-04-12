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


def process_diagnosis_with_multimodal(args, eps=1e-6):
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cxr_metafile = os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-metadata.csv')
    cxr_labelfile = os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-chexpert.csv')
    note_file = os.path.join(args.note_path, 'note_all.csv')
    all_stayfile = os.path.join(args.root_path, 'all_stays.csv')
    admissions_file = os.path.join(args.mimic4_path, 'hosp/admissions.csv')

    cxr_metadata = pd.read_csv(cxr_metafile)
    cxr_label = pd.read_csv(cxr_labelfile)
    note = pd.read_csv(note_file)
    all_stay = pd.read_csv(all_stayfile)
    admissions = pd.read_csv(admissions_file)

    # 选择AP类型CXR
    cxr_data = cxr_metadata[cxr_metadata['ViewPosition'] == 'AP']

    cxr_data.loc[:, 'StudyTime'] = cxr_data['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
    cxr_data.loc[:, 'StudyDateTime'] = pd.to_datetime(
        cxr_data['StudyDate'].astype(str) + ' ' + cxr_data['StudyTime'].astype(str), format="%Y%m%d %H%M%S")

    # 与EHR合并 为CXR匹配stay_id
    cxr_ehr_data = cxr_data[['subject_id', 'study_id', 'dicom_id', 'StudyDateTime']].merge(
        all_stay[['subject_id', 'stay_id', 'intime', 'outtime']], how='inner', on=['subject_id'])
    cxr_ehr_data.loc[:, 'intime'] = pd.to_datetime(cxr_ehr_data['intime'])
    cxr_ehr_data.loc[:, 'outtime'] = pd.to_datetime(cxr_ehr_data['outtime'])

    cxr_ehr_data = cxr_ehr_data.loc[
        (cxr_ehr_data.StudyDateTime >= cxr_ehr_data.intime) & ((cxr_ehr_data.StudyDateTime <= cxr_ehr_data.outtime))]

    cxr_ehr_data.loc[:, 'period_len'] = (cxr_ehr_data.StudyDateTime - cxr_ehr_data.intime).dt.total_seconds() / 3600
    cxr_ehr_data.loc[:, 'period_len'] = cxr_ehr_data['period_len'].round(6)
    cxr_ehr_data.loc[cxr_ehr_data['period_len'] <= 4, 'period_len'] = np.nan

    # 建立stay_id->ts file映射关系
    st_map = dict()
    patients = list(set(cxr_ehr_data['subject_id'].astype(str).unique())&set(filter(str.isdigit, os.listdir(args.root_path))))
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


                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]

                # Prevents time series data not start at hour 0
                first_time = event_times[0]
                cxr_ehr_data.loc[(cxr_ehr_data['stay_id'] == float(icustay)) & (cxr_ehr_data['period_len'] < first_time), 'period_len'] = np.nan

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                if os.path.isfile(os.path.join(args.ehr_path, output_ts_filename)) is not True:
                    continue

                st_map[icustay] = output_ts_filename

    cxr_ehr_data.loc[:, 'time_series'] = cxr_ehr_data['stay_id'].map(st_map)
    cxr_ehr_data.loc[cxr_ehr_data['period_len'].isnull(), 'time_series'] = np.nan
    cxr_ehr_data.loc[cxr_ehr_data['time_series'].isnull(), 'period_len'] = np.nan


    cxr_hadm_data = cxr_data[['subject_id', 'study_id', 'StudyDateTime']].merge(admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']], how='inner', on='subject_id')
    cxr_hadm_data = cxr_hadm_data.loc[(cxr_hadm_data.StudyDateTime >= cxr_hadm_data.admittime) & ((cxr_hadm_data.StudyDateTime <= cxr_hadm_data.dischtime))]
    cxr_note_data = cxr_hadm_data[['subject_id', 'study_id', 'hadm_id']].merge(note[['hadm_id', 'past_medical_history']], how='inner', on='hadm_id')

    all_data = cxr_data[['subject_id', 'study_id', 'dicom_id']].merge(cxr_ehr_data[['subject_id', 'time_series', 'period_len', 'dicom_id']], how='left', on=['subject_id', 'dicom_id'])
    all_data = all_data.merge(cxr_note_data[['subject_id', 'study_id', 'past_medical_history']], how='left', on=['subject_id', 'study_id'])
    all_data = all_data[['subject_id', 'study_id', 'time_series', 'period_len', 'dicom_id', 'past_medical_history']]
    all_data = all_data[~all_data.duplicated('study_id', keep='last')]
    all_data.to_csv(os.path.join(output_dir, "listfile.csv"), index=False)

    cxr_label = cxr_label.fillna(-2)
    final_data = all_data.merge(cxr_label, how='inner', on=['subject_id', 'study_id'])
    final_data.to_csv(os.path.join(output_dir, "multimodal_listfile.csv"), index=False)

    return cxr_ehr_data


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

    process_diagnosis_with_multimodal(args)

    # split into tr/val/te set
    create_train_val_test_set(args)


if __name__ == '__main__':
    main()
