import pandas as pd
import os
import random
import re
import string


def merge_multimodal_data(args, task):
    if task == 'length-of-stay' or task == 'decompensation':
        ehr_listfile = os.path.join(args.output_path, 'listfile_sampled.csv')
    else:
        ehr_listfile = os.path.join(args.output_path, 'listfile.csv')
    cxr_metafile = os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-metadata.csv')
    note_file = os.path.join(args.note_path, 'note_all.csv')
    all_stayfile = os.path.join(args.root_path, 'all_stays.csv')

    ehr_list = pd.read_csv(ehr_listfile)
    cxr_metadata = pd.read_csv(cxr_metafile)
    note = pd.read_csv(note_file)
    icu_stay_metadata = pd.read_csv(all_stayfile)

    columns = ['stay_id', 'intime', 'outtime']

    # Merge EHR time series data and icustay
    ehr_merged_icustays = ehr_list.merge(icu_stay_metadata[columns], how='inner', on='stay_id')

    # Determine different time ranges for different tasks (intime/endtime)
    ehr_merged_icustays.intime = pd.to_datetime(ehr_merged_icustays.intime)
    ehr_merged_icustays.outtime = pd.to_datetime(ehr_merged_icustays.outtime)
    if task == 'in-hospital-mortality':
        ehr_merged_icustays['endtime'] = ehr_merged_icustays['intime'] + ehr_merged_icustays['period_length'].apply(
            lambda x: pd.DateOffset(hours=48))
    elif task == 'decompensation' or task == 'length-of-stay':
        ehr_merged_icustays['endtime'] = ehr_merged_icustays['intime'] + ehr_merged_icustays['period_length'].apply(
            lambda x: pd.DateOffset(hours=x))
    else:
        ehr_merged_icustays['endtime'] = ehr_merged_icustays.outtime

    # Merge EHR time series and CXR. Obtain the corresponding jpg filename and timestamp
    ehr_cxr_merged = ehr_merged_icustays.merge(cxr_metadata, how='inner', on='subject_id')
    ehr_cxr_merged['StudyTime'] = ehr_cxr_merged['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
    ehr_cxr_merged['StudyDateTime'] = pd.to_datetime(
        ehr_cxr_merged['StudyDate'].astype(str) + ' ' + ehr_cxr_merged['StudyTime'].astype(str), format="%Y%m%d %H%M%S")

    end_time = ehr_cxr_merged.endtime

    # Delete the EHR-CXR pairs that do not match the time range
    ehr_cxr_merged_during = ehr_cxr_merged.loc[
        (ehr_cxr_merged.StudyDateTime >= ehr_cxr_merged.intime) & ((ehr_cxr_merged.StudyDateTime <= end_time))]

    # Select CXR of the AP type
    ehr_cxr_merged_AP = ehr_cxr_merged_during[ehr_cxr_merged_during['ViewPosition'] == 'AP']

    # Select the last CXR of the sample
    ehr_cxr_merged_sorted = ehr_cxr_merged_AP.sort_values(by=['time_series', 'period_length', 'StudyDateTime'],
                                                          ascending=[True, True, True])
    ehr_cxr_merged_final = ehr_cxr_merged_sorted.drop_duplicates(subset=['time_series', 'period_length'], keep='last')

    # Merged with the original EHR to obtain the full dataset of partially CXR-free modality
    all_merged = ehr_list.merge(ehr_cxr_merged_final[['time_series', 'period_length', 'dicom_id']], how='outer',
                                on=['time_series', 'period_length'])

    # Merge Note data and icustay
    note_stay = icu_stay_metadata[['subject_id', 'hadm_id', 'stay_id']].merge(note, how='inner', on=['subject_id', 'hadm_id'])
    all_merged = all_merged.merge(note_stay[['stay_id','past_medical_history']], how='left', on='stay_id')

    # Adjust column order
    df_id = all_merged.dicom_id
    all_merged = all_merged.drop('dicom_id', axis=1)
    all_merged.insert(4, 'dicom_id', df_id)

    df_id = all_merged.past_medical_history
    all_merged = all_merged.drop('past_medical_history', axis=1)
    all_merged.insert(5, 'past_medical_history', df_id)

    all_merged.to_csv(os.path.join(args.output_path, 'multimodal_listfile.csv'), index=False)

    return all_merged


# Remove symbols and line breaks from text
def remove_symbol(text):
    text = text.replace('\n', '')

    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i, '')

    return text


# extract Brief Hospital Course in the discharge summary
def extract_BHC(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern1 = re.compile(r"brief hospital course:(.*?)medications on admission", re.DOTALL)
    pattern2 = re.compile(r"brief Hospital Course:(.*?)discharge medications", re.DOTALL)

    if "brief hospital course:" in text:
        if re.search(pattern1, text):
            match = re.search(pattern1, text).group(1).strip()
        elif re.search(pattern2, text):
            match = re.search(pattern2, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Chief Complaint in the discharge summary
def extract_CC(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"chief complaint:(.*?)major surgical or invasive procedure", re.DOTALL)

    if "chief complaint:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Past Medical History in the discharge summary
def extract_PMH(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"past medical history:(.*?)social history", re.DOTALL)

    if "past medical history:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Medications on Admission in the discharge summary
def extract_MA(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"medications on admission:(.*?)discharge medications", re.DOTALL)

    if "medications on admission:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# Extract required data from raw MIMIC-NOTE
def process_note(note_path, output_path):
    new_df = pd.read_csv(os.path.join(note_path, 'note/discharge.csv'))

    # Extract Brief Hospital Course part as the overall admission summary (For DRG task)
    new_df['brief_hospital_course'] = new_df['text'].apply(extract_BHC)

    # Extract Past Medical History  (For 6 tasks)
    new_df['past_medical_history'] = new_df['text'].apply(extract_PMH)

    # Extract Chief Complaint and Medications on Admission
    # new_df['chief_complaint'] = new_df['text'].apply(extract_CC)
    # new_df['medications_on_admission'] = new_df['text'].apply(extract_MA)

    new_df.drop(['text', 'note_id', 'note_type', 'note_seq'], axis=1, inplace=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    new_df.to_csv(os.path.join(output_path, 'note_all.csv'), index=False)


# split all patients into train/val/test sets
def split_train_val_test_id(admissions):
    all_id = list(admissions['subject_id'].unique())

    test_nums = int(len(all_id) * 0.2)
    val_nums = int(len(all_id) * 0.1)
    split_choose = [2] * test_nums + [1] * val_nums + [0] * (len(all_id) - test_nums - val_nums)
    random.seed(42)
    random.shuffle(split_choose)

    tvt_set = pd.DataFrame([all_id, split_choose])
    tvt_set.T.to_csv(os.path.join(os.path.dirname(__file__), '../resources/tvtset.csv'), index=False, header=False)


#
def create_train_val_test_set(args):
    train_patients = set()
    val_patients = set()
    test_patients = set()
    with open(os.path.join(os.path.dirname(__file__), '../resources/tvtset.csv'), 'r') as tvtset_file:
        for line in tvtset_file:
            x, y = line.split(',')
            if int(y) == 2:
                test_patients.add(x)
            elif int(y) == 1:
                val_patients.add(x)
            else:
                train_patients.add(x)

    with open(os.path.join(args.output_path, 'multimodal_listfile.csv')) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        lines = lines[1:]

    train_lines = [x for x in lines if x.split(',')[0] in train_patients]
    val_lines = [x for x in lines if x.split(',')[0] in val_patients]
    test_lines = [x for x in lines if x.split(',')[0] in test_patients]

    # assert len(train_lines) + len(val_lines) + len(test_lines)== len(lines)

    with open(os.path.join(args.output_path, 'train_multimodal_listfile.csv'), 'w') as train_listfile:
        train_listfile.write(header)
        for line in train_lines:
            train_listfile.write(line)

    with open(os.path.join(args.output_path, 'val_multimodal_listfile.csv'), 'w') as val_listfile:
        val_listfile.write(header)
        for line in val_lines:
            val_listfile.write(line)

    with open(os.path.join(args.output_path, 'test_multimodal_listfile.csv'), 'w') as test_listfile:
        test_listfile.write(header)
        for line in test_lines:
            test_listfile.write(line)

    print('Train samples:', len(train_lines), ', Val samples:', len(val_lines), ', Test samples:', len(test_lines))


# Divide labels of length-of-stay into 10 categories
def get_bin_custom(x, nbins=10):
    inf = 10e6
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]

    for i in range(nbins):
        a = bins[i][0] * 24.0
        b = bins[i][1] * 24.0
        if a <= x < b:
            return i
    return None


def random_sample(args):
    ehr_listfile = os.path.join(args.output_path, "listfile.csv")
    ehr_list = pd.read_csv(ehr_listfile)

    # task 'length-of-stay' needs to transform labels
    # if task == 'length-of-stay':
    #     ehr_list['y_true'] = ehr_list['y_true'].apply(get_bin_custom)

    shuffled_ehr_list = ehr_list.sample(frac=1, random_state=42)
    new_ehr_list = shuffled_ehr_list.drop_duplicates(['stay_id'], keep='first')

    new_ehr_list.to_csv(os.path.join(args.output_path, "listfile_sampled.csv"), index=False)