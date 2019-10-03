import sys
import os
import cv2
import glob
import zipfile
import argparse
import requests
from shutil import rmtree


"""
Source : https://github.com/ZZUTK/SRNTT/blob/master/download_dataset.py
Code changed by Devkihyun.
"""


CUFED5_TEST_DATA_URL = 'https://drive.google.com/uc?export=download&id=1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph'
DIV2K_INPUT_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=1nGeoNLVd-zPifH6sLOYvpY9lVYKnUc0w'
DIV2K_REF_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=1sj72-zL3cGjsVqbbnk3PxJxjQWATQx61'
CUFED_INPUT_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=1gN5IPZgPNkjdeXdTySe1Urog5OG8mrLc'
CUFED_REF_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=13BX-UY4jUZu9S--X2Cd6yZ-3nH77nqo_'


datasets = {
    'CUFED5': {'name': 'CUFED5', 'url': CUFED5_TEST_DATA_URL, 'save_dir': 'data/test', 'data_size': 233, 'img_size': None},
    'DIV2K_input': {'name': 'DIV2K_input', 'url': DIV2K_INPUT_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 1835, 'img_size': 320},
    'DIV2K_ref': {'name': 'DIV2K_ref', 'url': DIV2K_REF_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 1905, 'img_size': 320},
    'CUFED_input': {'name': 'CUFED_input', 'url': CUFED_INPUT_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 567, 'img_size': 160},
    'CUFED_ref': {'name': 'CUFED_ref', 'url': CUFED_REF_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 588, 'img_size': 160}
}


def download_file_from_google_drive(url, save_dir, data_name, data_size=None):
    if not os.path.exists(os.path.join(save_dir, '%s.zip' % data_name)):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with requests.Session() as session:
            response = session.get(url, stream=True)
            token = get_confirm_token(response)
            if token:
                response = session.get(url, params={'confirm': token}, stream=True)
            save_response_content(response, os.path.join(save_dir, '%s.zip' % data_name), data_size)
    else:
        print('[!] %s already exist! Skip download.' % os.path.join(save_dir, '%s.zip' % data_name))

    if os.path.exists(os.path.join(save_dir, data_name.split('_')[-1])):
        rmtree(os.path.join(save_dir, data_name.split('_')[-1]))

    zip_ref = zipfile.ZipFile(os.path.join(save_dir, '%s.zip' % data_name), 'r')
    if 'train' in save_dir:
        print('>> Unzip %s --> %s' % (os.path.join(save_dir, '%s.zip' % data_name),
                                      os.path.join(save_dir, data_name.split('_')[0], data_name.split('_')[-1])))
        zip_ref.extractall(os.path.join(save_dir, data_name.split('_')[0]))
    else:
        print('>> Unzip %s --> %s' % (os.path.join(save_dir, '%s.zip' % data_name),
                                      os.path.join(save_dir, data_name.split('_')[-1])))
        zip_ref.extractall(save_dir)
    zip_ref.close()
    os.remove(os.path.join(save_dir, '%s.zip' % data_name))


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, save_dir, data_size=None):
    chunk_size = 1024 * 1024  # in byte
    with open(save_dir, "wb") as f:
        len_content = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                len_content += len(chunk)
                if data_size is not None:
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (save_dir, min(len_content / 1024. / 1024. / data_size * 100, 100)))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('\r>> Downloading %s %.1f MB' % (save_dir, len_content / 1024. / 1024.))
                    sys.stdout.flush()
        print('')
        

def check_image_size(save_path, img_size):
    get_image = lambda path : cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    path_list = glob.glob(save_path)

    count = 0
    for path in path_list:
        image = get_image(path)
        size = image.shape[:2]
        
        if size != (img_size,img_size):
            print(f'>> {path} -> size = {size}')
            resize_image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            resize_image[:size[0], :size[1],:] = image
            resize_image = cv2.cvtColor(resize_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, resize_image)
            count += 1
        
    return count
    

if __name__ == "__main__":
    is_downloaded = False
    for key in datasets:
        dataset = datasets[key]

        url       = dataset['url']
        save_dir  = dataset['save_dir']
        data_name = dataset['name']
        data_size = dataset['data_size']
        img_size  = dataset['img_size']
        
        download_file_from_google_drive(
            url=url,
            save_dir=save_dir,
            data_name=data_name,
            data_size=data_size
        )
        
        if key != 'CUFED5':
            print(f">> Check abnormal image in {data_name}")
            save_path = os.path.join(save_dir, data_name.split('_')[0], data_name.split('_')[1], '*')
            abnomal_count = check_image_size(save_path, img_size)
            
            print(f">> Abnormal size count of {key}: {abnomal_count}")
        