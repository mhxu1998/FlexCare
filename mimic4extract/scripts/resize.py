# import thread
import time
from PIL import Image
import glob
from tqdm import tqdm
import os
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import shutil


def resize_images(path):
    basewidth = 512
    filename = path.split('/')[-1]
    img = Image.open(path)

    wpercent = (basewidth / float(img.size[0]))

    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize))

    img.save(f'{args.cxr_path}/resized/{filename}')


parser = argparse.ArgumentParser(description="Resize the CXR image.")
parser.add_argument('mimic_cxr_path', type=str, default='mimic-cxr-jpg/2.0.0', help="Path to raw MIMIC-CXR folder.")
parser.add_argument('cxr_path', type=str, default='data/cxr', help="Directory where the processed image should be stored.")
args, _ = parser.parse_known_args()

print('starting process CXR images.')

if not os.path.exists(os.path.join(args.cxr_path, 'resized')):
    os.makedirs(os.path.join(args.cxr_path, 'resized'))

paths_all = glob.glob(os.path.join(args.mimic_cxr_path, 'files/**/*.jpg'), recursive=True)
print('all', len(paths_all))

paths_done = glob.glob(os.path.join(args.cxr_path, 'resized/*.jpg'), recursive = True)
print('done', len(paths_done))

done_files = [os.path.basename(path) for path in paths_done]
paths = [path for path in paths_all if os.path.basename(path) not in done_files ]
print('left', len(paths))




threads = 10

for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i: i + threads]
    pool = ThreadPool(len(paths_subset))
    pool.map(resize_images, paths_subset)
    pool.close()
    pool.join()

# copy mimic-cxr-2.0.0-metadata.csv and mimic-cxr-2.0.0-chexpert.csv to the new cxr folder
if not os.path.exists(os.path.join(args.mimic_cxr_path, 'mimic-cxr-2.0.0-metadata.csv')):
    print('There is no file: mimic-cxr-2.0.0-metadata.csv')
else:
    shutil.copy(os.path.join(args.mimic_cxr_path, 'mimic-cxr-2.0.0-metadata.csv'), os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-metadata.csv'))

if not os.path.exists(os.path.join(args.mimic_cxr_path, 'mimic-cxr-2.0.0-chexpert.csv')):
    print('There is no file: mimic-cxr-2.0.0-chexpert.csv')
else:
    shutil.copy(os.path.join(args.mimic_cxr_path, 'mimic-cxr-2.0.0-chexpert.csv'), os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-chexpert.csv'))