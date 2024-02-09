from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import yaml
import random
random.seed(49297)
from tqdm import tqdm
from mimic4extract.scripts.utils import merge_multimodal_data, create_train_val_test_set

def process_time_series_with_label(args, definitions, code_to_group, id_to_group, group_to_id, eps=1e-6):
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
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

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                if os.path.isfile(os.path.join(args.ehr_path, output_ts_filename)) is not True:
                    continue

                cur_labels = [0 for i in range(len(id_to_group))]

                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"), dtype={"icd_code": str})
                diagnoses_df = diagnoses_df[diagnoses_df.stay_id == icustay]
                for index, row in diagnoses_df.iterrows():
                    if row['USE_IN_BENCHMARK']:
                        code = row['icd_code']
                        if code in code_to_group:
                            group = code_to_group[code]
                            group_id = group_to_id[group]
                            cur_labels[group_id] = 1
                        # else:
                        #     print(f'{code} code not found')
                # import pdb; pdb.set_trace()
                cur_labels = [x for (i, x) in enumerate(cur_labels) if definitions[id_to_group[i]]['use_in_benchmark']]

                # xty_triples.append((output_ts_filename, los, icustay, cur_labels))
                xty_triples.append((patient, icustay, output_ts_filename, los, cur_labels))

    print("Number of created samples:", len(xty_triples))
    xty_triples = sorted(xty_triples)

    codes_in_benchmark = [x for x in id_to_group if definitions[x]['use_in_benchmark']]

    listfile_header = "subject_id,stay_id,time_series,period_length," + ",".join(codes_in_benchmark)
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (patient, icustay, ts_filename, t, y) in xty_triples:
            labels = ','.join(map(str, y))
            listfile.write('{},{},{},{:.6f},{}\n'.format(patient, icustay, ts_filename, t, labels))


def main():
    parser = argparse.ArgumentParser(description="Create data for phenotype classification task.")
    parser.add_argument('root_path', type=str, help="Path to root folder.")
    parser.add_argument('ehr_path', type=str, help="Path to time series data folder.")
    parser.add_argument('cxr_path', type=str, help="Path to cxr data folder.")
    parser.add_argument('note_path', type=str, help="Path to note data folder.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/icd_9_10_definitions_2.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.safe_load(definitions_file)

    code_to_group = {}

    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group
            else:
                print(f'code, {code}')
                assert code_to_group[code] == group

    # import pdb;pdb.set_trace()
    # ['Diabetes mellitus with complication', ]
    # 'ICD-10-CM CODE' 'Default CCSR CATEGORY DESCRIPTION IP'
    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_time_series_with_label(args, definitions, code_to_group, id_to_group, group_to_id)

    # merge multimodal data
    merge_multimodal_data(args, 'phenotyping')

    # split into tr/val/te set
    create_train_val_test_set(args)


if __name__ == '__main__':
    main()
