import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Multimodal_dataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, ehr_dir, cxr_dir, task, transform=None, return_names=True, period_length=48.0):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.period_length = period_length
        self.task = task

        self.ehr_dir = ehr_dir
        self.cxr_dir = cxr_dir
        # self.note_dir = note_dir

        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[6:]
        self._data = self._data[1:]

        self.transform = transform

        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[1]+'_time_'+str(mas[3]): {
                'subject_id': float(mas[0]),
                'stay_id': float(mas[1]),
                'ehr_file': str(mas[2]),
                'time': str(mas[3]),
                'cxr_id': str(mas[4]),
                'note': str(mas[5]),
                'labels': list(map(float, mas[6:])),
            }
            for mas in self._data
        }

        self.names = list(self.data_map.keys())

    def _read_timeseries(self, ts_filename, time_bound=None):
        ret = []
        with open(os.path.join(self.ehr_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return np.stack(ret), header

    def _read_cxr(self, cxr_filename):
        if cxr_filename != '':
            file_path = f'{self.cxr_dir}/resized/'+cxr_filename+'.jpg'
            img = Image.open(file_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = None

        return img

    def read_by_file_name(self, index, time_bound=None):
        ehr_file = self.data_map[index]['ehr_file']
        t = self.data_map[index]['time'] if time_bound is None else time_bound
        t = float(t) if t!='' else -1
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        cxr_id = self.data_map[index]['cxr_id']
        note = self.data_map[index]['note']
        if self.task in ['decompensation', 'length-of-stay', 'diagnosis']:
            time_bound = t

        if ehr_file=='':
            (X, header) = None, None
        else:
            (X, header) = self._read_timeseries(ehr_file, time_bound=time_bound)

        img = self._read_cxr(cxr_id)

        return {"X": X,
                "t": t,
                "y": y,
                "img": img,
                "note": note,
                'stay_id': stay_id,
                "header": header,
                "name": index}

    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self.period_length
        img = ret["img"]
        note = ret["note"]
        ys = ret["y"]
        names = ret["name"]

        if data is not None:
            data = self.discretizer.transform(data, end=ts)[0]
            if self.normalizer is not None:
                data = self.normalizer.transform(data)
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, img, note, ys, index, ret, self.task

    def __len__(self):
        return len(self.names)


def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = []
    train_transforms.append(transforms.Resize(256))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)

    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))
    test_transforms.append(transforms.CenterCrop(args.crop))
    test_transforms.append(transforms.ToTensor())
    test_transforms.append(normalize)

    return train_transforms, test_transforms


def get_multimodal_datasets(discretizer, normalizer, args, task):
    train_transforms, test_transforms = get_transforms(args)

    train_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/train_multimodal_listfile.csv',
                          args.ehr_path, args.cxr_path, task, transforms.Compose(train_transforms))
    val_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/val_multimodal_listfile.csv',
                        args.ehr_path, args.cxr_path, task, transforms.Compose(test_transforms))
    test_ds = Multimodal_dataset(discretizer, normalizer, f'{args.data_path}/{task}/test_multimodal_listfile.csv',
                         args.ehr_path, args.cxr_path, task, transforms.Compose(test_transforms))
    return train_ds, val_ds, test_ds



