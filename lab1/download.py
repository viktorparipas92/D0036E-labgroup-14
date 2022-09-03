import os
import tarfile
import urllib.request


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'


def fetch_data_from_tgz(path, download_root=DOWNLOAD_ROOT):
    os.makedirs(path, exist_ok=True)
    filename = path.split('/')[-1]
    tgz_path = os.path.join(path, f'{filename}.tgz')
    url = f'{download_root}{tgz_path}'
    urllib.request.urlretrieve(url, tgz_path)
    tgz_file = tarfile.open(tgz_path)
    tgz_file.extractall(path=path)
    tgz_file.close()