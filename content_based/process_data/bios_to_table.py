import glob
import sys
import os

def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(round(completed,2)) + ' %' )
            sys.stdout.flush()

def bios_to_table(fakeDataset):

    dataset_path = '../fake_dataset/' if fakeDataset else '../dataset/'  
    bios_path = dataset_path +'bios/'
    file_output = dataset_path +'bios.txt'

    # os.remove(file_output)

    f = open(file_output, 'w+')
    f.write('id\tbio\n')
    bios = glob.glob(bios_path+'*')

    i = 0
    completed =0
    print('Reading ' +str(len(bios))+ ' bios...')

    for artistid in bios:
        i = i+1
        print_progress('',completed,i/len(bios) * 100)
        with open(artistid) as ff:

            f.write(artistid.replace(bios_path,'')[:-4] + '\t'+ff.read().replace('\n','').replace('\t','')+'\n')
    print(" Done")