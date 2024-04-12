import numpy as np
import sys

from arguments import args_parser
parser = args_parser()
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
import datetime
import random

from torch.utils.data import Dataset, DataLoader
from utils import Discretizer, Normalizer, my_metrics, is_ascending
from dataset.dataloader import get_multimodal_datasets
from mymodel.model import FlexCare

torch.autograd.set_detect_anomaly(True)
# torch.use_deterministic_algorithms(True)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

num_workers = args.num_workers
adjust_step = 2

if args.device != "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
args.task = args.task.split(',')

task_weight = {'in-hospital-mortality':0.2,
               'length-of-stay':0.5,
                'phenotyping':1,
               'decompensation':0.2,
                'readmission':0.2,
                'diagnosis':0.2}

# Process mini batch input
def my_collate(batch):
    # Time series data  (When missing, use an all-zero vector with shape of [1,76])
    ehr = [item[0][-512:] if np.array_equal(item[0], None) is False else np.zeros((1,76)) for item in batch]
    ehr, ehr_length = pad_zeros(ehr)
    mask_ehr = np.array([1 if np.array_equal(item[0], None) is False else 0 for item in batch])     # Marks whether EHR is included
    ehr_length = [0 if mask_ehr[i] == 0 else ehr_length[i] for i in range(len(ehr_length))]  # Remove fictitious time series
    
    # CXR image data    (When missing, use an all-zero vector with shape of [3,224,224])
    cxr = torch.stack([item[1] if item[1] != None else torch.zeros(3, 224, 224) for item in batch])
    mask_cxr = np.array([1 if item[1] != None else 0 for item in batch])

    # Note text data    (An empty string has been used to indicate modality missing)
    note = [item[2] for item in batch]
    mask_note = np.array([1 if item[2] != '' else 0 for item in batch])

    # Label
    label = np.array([item[3] for item in batch]).reshape(len(batch),-1)

    # Task
    replace_dict = {'in-hospital-mortality':0, 'decompensation':1, 'phenotyping':2, 'length-of-stay':3, 'readmission':4, 'diagnosis':5, 'drg':6}
    task_index = np.array([replace_dict[item[6]] if item[6] in replace_dict else -1 for item in batch])

    return [ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index]


# Pad the time series to the same length
def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in ret]
    return np.array(ret), seq_length


def read_timeseries(path):
    path = f'{args.ehr_path}/10002430_episode1_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)




def main():
    # Time series data discretization
    discretizer = Discretizer(timestep=float(args.timestep), store_masks=True, impute_strategy='previous', start_time='zero')
    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    # Mutli-task dataloader
    mutli_train_dl = []
    mutli_val_dl = []
    mutli_test_dl = []
    for task in args.task:
        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = 'normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(1.0)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)

        train_ds, val_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, task)
        mutli_train_dl.append(DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=num_workers, drop_last=True))
        mutli_val_dl.append(DataLoader(val_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=num_workers, drop_last=False))
        mutli_test_dl.append(DataLoader(test_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=num_workers, drop_last=False))

    model = FlexCare(hidden_dim=args.hidden_dim, device=device).to(device)

    criterion = torch.nn.BCELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    multi_lr = [args.lr for i in range(len(args.task))]

    file_path = 'log/['+args.model+']_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '_epoch_' + str(args.epochs) +'_' + str(args.task)+ '_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.txt'
    best_epoch = 0
    best_valid_res = 0
    best_test_auc = 0
    best_test_aupr = 0

    val_record = [[] for i in range(len(args.task))]

    if not os.path.exists('checkpoits'):
        os.mkdir('checkpoits')

    if not os.path.exists('log'):
        os.mkdir('log')


    for epoch in tqdm(range(1, args.epochs+1)):
        print('Epoch:', epoch)
        # Train
        model.train()
        train_loss = 0

        task_list = list(range(len(mutli_train_dl)))
        print(multi_lr)
        for t in range(len(mutli_train_dl)):
            task_now = args.task[t]
            print('Task:', task_now, ' Training!')

            if 'our' in args.model and len(args.task) > 1:
                optimizer.param_groups[0]['lr'] = multi_lr[t]

            with tqdm(mutli_train_dl[t], position=0, ncols=150, colour='#666666') as tqdm_range:
                for i, data in enumerate(tqdm_range):
                    if task_now == 'drg' and i>len(mutli_train_dl[t])/100:
                        break

                    optimizer.zero_grad()
                    ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                    ehr = torch.from_numpy(ehr).float().to(device)
                    cxr = cxr.to(device)
                    mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                    mask_cxr = torch.from_numpy(mask_cxr).long().to(device)
                    mask_note = torch.from_numpy(mask_note).long().to(device)
                    y_true = torch.from_numpy(label).float().to(device)
                    task_index = torch.from_numpy(task_index).long().to(device)

                    y_pred = model(ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, task_index)

                    # Choose loss function based on the task
                    if task_now in ['length-of-stay','drg']:
                        criterion_now = criterion_ce
                        y_true = y_true.long().view(-1)
                    else:
                        criterion_now = criterion

                    y_pred, ortho_loss, moe_loss = y_pred


                    if task_now == 'diagnosis':
                        label_mask = (y_true > -1)
                        y_pred2 = y_pred.clone()
                        y_true2 = y_true.clone()
                        y_pred = y_pred[label_mask]
                        y_true = y_true[label_mask]
                        loss = criterion_now(y_pred, y_true)
                        y_pred = y_pred2
                        y_true = y_true2
                    else:
                        loss = criterion_now(y_pred, y_true)

                    loss = (loss + ortho_loss*0.5 + moe_loss)*task_weight[task_now]

                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

        print('Train loss', train_loss)

        with open(file_path, "a", encoding='utf-8') as f:
            f.write('Epoch:' + str(epoch) + '\n')
            f.write('lr:'+str(multi_lr)+ '\n')
            f.write('Train loss:'+str(train_loss)+'\n')


        with torch.no_grad():
            # Valid
            model.eval()
            valid_loss = 0
            valid_res = 0

            for t in range(len(mutli_val_dl)):
                task_now = args.task[t]
                task_val_loss = 0
                with tqdm(mutli_val_dl[t], position=0, ncols=150, colour='#666666') as tqdm_range:
                    outGT = torch.FloatTensor().to(device)
                    outPRED = torch.FloatTensor().to(device)
                    for i, data in enumerate(tqdm_range):
                        ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                        ehr = torch.from_numpy(ehr).float().to(device)
                        cxr = cxr.to(device)
                        mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                        mask_cxr = torch.from_numpy(mask_cxr).long().to(device)
                        mask_note = torch.from_numpy(mask_note).long().to(device)
                        y_true = torch.from_numpy(label).float().to(device)
                        task_index = torch.from_numpy(task_index).long().to(device)

                        y_pred = model(ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, task_index)

                        # Choose loss function based on the task
                        if task_now in ['length-of-stay','drg']:
                            criterion_now = criterion_ce
                            y_true = y_true.long().view(-1)
                        else:
                            criterion_now = criterion

                        y_pred = y_pred.reshape(ehr.shape[0], -1)
                        if task_now == 'diagnosis':
                            label_mask = (y_true > -1)
                            y_pred2 = y_pred.clone()
                            y_true2 = y_true.clone()
                            y_pred = y_pred[label_mask]
                            y_true = y_true[label_mask]
                            loss = criterion_now(y_pred, y_true)
                            y_pred = y_pred2
                            y_true = y_true2
                        else:
                            loss = criterion_now(y_pred, y_true)

                        valid_loss += loss.item()
                        task_val_loss += loss.item()

                        if task_now in ['length-of-stay','drg']:
                            _, y_pred = torch.max(y_pred, dim=1)

                        outPRED = torch.cat((outPRED, y_pred), 0)
                        outGT = torch.cat((outGT, y_true), 0)

                val_record[t].append(task_val_loss)

                auc, aupr = my_metrics(outGT, outPRED, task_now)
                valid_res += (auc + aupr)

                print('Task: ', task_now, ' Valid AUC:', auc, '   AUPR:', aupr, '   Loss:', task_val_loss)

                with open(file_path, "a", encoding='utf-8') as f:
                    f.write('Task: '+task_now+' Valid AUC:'+str(auc)+'   AUPR:'+str(aupr)+'   Loss:'+str(task_val_loss) + '\n')

            print('Valid loss:', valid_loss, 'Valid res:', valid_res)
            with open(file_path, "a", encoding='utf-8') as f:
                f.write('Valid loss:' + str(valid_loss)+'Valid res:' + str(valid_res)+'\n')

            if valid_res > best_valid_res:
                best_epoch = epoch
                best_valid_res = valid_res

            for i in range(len(val_record)):
                if (len(val_record[i]) < adjust_step):
                    continue
                else:
                    if is_ascending(val_record[i][-adjust_step:]):
                        multi_lr[i] = multi_lr[i]/2

            # Test
            test_loss = 0
            test_auc = []
            test_aupr = []
            for t in range(len(mutli_test_dl)):
                task_now = args.task[t]
                if task_now == 'drg':
                    break
                with tqdm(mutli_test_dl[t], position=0, ncols=150, colour='#666666') as tqdm_range:
                    outGT = torch.FloatTensor().to(device)
                    outPRED = torch.FloatTensor().to(device)
                    for i, data in enumerate(tqdm_range):
                        ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, label, task_index = data
                        ehr = torch.from_numpy(ehr).float().to(device)
                        cxr = cxr.to(device)
                        mask_ehr = torch.from_numpy(mask_ehr).long().to(device)
                        mask_cxr = torch.from_numpy(mask_cxr).long().to(device)
                        mask_note = torch.from_numpy(mask_note).long().to(device)
                        y_true = torch.from_numpy(label).float().to(device)
                        task_index = torch.from_numpy(task_index).long().to(device)

                        y_pred = model(ehr, ehr_length, mask_ehr, cxr, mask_cxr, note, mask_note, task_index)

                        # Choose loss function based on the task
                        if task_now in ['length-of-stay','drg']:
                            criterion_now = criterion_ce
                            y_true = y_true.long().view(-1)
                        else:
                            criterion_now = criterion

                        y_pred = y_pred.reshape(ehr.shape[0], -1)
                        if task_now == 'diagnosis':
                            label_mask = (y_true > -1)
                            y_pred2 = y_pred.clone()
                            y_true2 = y_true.clone()
                            y_pred = y_pred[label_mask]
                            y_true = y_true[label_mask]
                            loss = criterion_now(y_pred, y_true)
                            y_pred = y_pred2
                            y_true = y_true2
                        else:
                            loss = criterion_now(y_pred, y_true)

                        test_loss += loss.item()

                        if task_now in ['length-of-stay','drg']:
                            _, y_pred = torch.max(y_pred, dim=1)

                        outPRED = torch.cat((outPRED, y_pred), 0)
                        outGT = torch.cat((outGT, y_true), 0)

                auc, aupr = my_metrics(outGT, outPRED, task_now)
                test_auc.append(auc)
                test_aupr.append(aupr)
                print('Task: ', task_now, ' Test AUC:', auc, '   AUPR:', aupr)

                with open(file_path, "a", encoding='utf-8') as f:
                    f.write('Task: '+task_now+' Test AUC:'+str(auc)+'   AUPR:'+str(aupr) + '\n')

            print('Test loss:', test_loss)
            with open(file_path, "a", encoding='utf-8') as f:
                f.write('Test loss:' + str(test_loss)+'\n')

            if best_epoch == epoch:
                best_test_auc = test_auc
                best_test_aupr = test_aupr
                if 'our' in args.model or 'drg' in args.task:
                    torch.save(model.state_dict(), 'checkpoints/'+file_path[3:-4]+'.pt')

    print('Best Epoch:', best_epoch)
    print('Best AUC:', best_test_auc, '     AUPR:', best_test_aupr)

    with open(file_path, "a", encoding='utf-8') as f:
        f.write('Best Epoch:' + str(best_epoch) + '  Best Val Res:' + str(best_valid_res) + '\n')
        for t in range(len(mutli_test_dl)):
            task_now = args.task[t]
            f.write('Task: ' + task_now + ' Best AUC:' + str(best_test_auc[t]) + '   AUPR:' + str(best_test_aupr[t]) + '\n')

if __name__ == '__main__':
    main()
