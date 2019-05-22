import urllib.request
import tarfile
import os

import sys
import time


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()




def downloadFile(url, path):
    urllib.request.urlretrieve(url, path, reporthook) 

def extractFile(path):
    print('Extracting ', path, end=' ')
    tar = tarfile.open(path, 'r:gz' )
    tar.extractall(pathDataset)
    tar.close()
    print('Extracted')

def getData(url, path, folder):
    if os.path.isfile(folder):
        print(path, ' already exists, skipping download')
    else:
        downloadFile(url, path)

    if  os.path.exists(folder):
        print('Skipping extract for ', folder, ', already exists')
    else: 
        extractFile(path)

pathDataset = './dataset/'
def getDataset():


    urlBios = 'https://zenodo.org/record/831348/files/bios_msd-a.tar.gz?download=1'
    nameBios = 'bios.tar.gz'
    folderBios = 'bios'

    urlTags = 'https://zenodo.org/record/831348/files/tags_msd-a.tar.gz?download=1'
    nameTags = 'tags.tar.gz'
    folderTags = 'tags'

    urlTriplets = 'https://zenodo.org/record/831348/files/triplets_msd-a.tar.gz?download=1'
    nameTriplets = 'triplets.tar.gz'
    folderTriplets = 'triplets'

    if os.path.exists(pathDataset):
        print(pathDataset)
        print('Dataset already exists')
        exit(0)
    os.mkdir(pathDataset)
    getData(urlBios, pathDataset+nameBios,pathDataset + folderBios)
    getData(urlTags, pathDataset+nameTags, pathDataset + folderTags)
    getData(urlTriplets, pathDataset+nameTriplets, pathDataset + folderTriplets)

# getDataset()



