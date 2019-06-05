
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def read_data(dataset_path, chunkSize, low,high,nbins, Nlines):
    counter = 0
    completed = 0

    bin_edges = np.linspace(low, high, nbins + 1)
    total = np.zeros(nbins, np.uint)

    # iterate over  dataset in chunks
    for chunk in pd.read_csv(dataset_path, sep='\t',header=None, chunksize=chunkSize):

        # compute bin counts over the 3rd column (plays)
        subtotal, e = np.histogram(chunk.iloc[:, 2], bins=bin_edges)

        # accumulate bin counts over chunks
        total += subtotal.astype(np.uint)

        # progress counter
        counter += chunkSize
        new_completed = int(round(float(counter)/Nlines * 100))
       
        print_progress('Reading data ... ', completed, new_completed)
    print(' ...completed')
    
    return bin_edges,total


def plot_histogram(bin_edgess, totall):

    print(len(bin_edgess))
    print(len(totall))


    plt.hist(bin_edgess[:-1], bins=bin_edgess, weights=totall, edgecolor='black')
    plt.xlabel('Number of plays ')
    plt.ylabel('Frequency')
    plt.title('Histogram of MSD-AG')
    plt.grid(True)
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()

def print_progress(text, completed, new_completed):
     if (new_completed > completed): 
            completed = new_completed
            sys.stdout.write('\r'+text+ str(completed) + ' %' )
            sys.stdout.flush()

def compute_boundaries(dataset_path, chunkSize, Nlines):
    low = 0
    high = -1

    counter = 0
    completed = 0

    for chunk in pd.read_csv(dataset_path, sep='\t',header=None, chunksize=chunkSize):
        high = np.maximum(chunk.iloc[:, 2].max(), high)
        
        # progress counter
        counter += chunkSize
        new_completed = int(round(float(counter)/Nlines * 100))
       
        print_progress('Computing boundaries... ', completed, new_completed)
    print(' ...completed')

    return low,high

def countLines(dataset_path):
    print('Counting lines')
    return sum(1 for line in open(dataset_path))
# Pre computed data
p_bin_edges = [0., 48.335, 96.67, 145.005, 193.34, 241.675, 290.01, 338.345,
  386.68,  435.015,  483.35,  531.685,  580.02,  628.355,  676.69,  725.025,
  773.36,  821.695,  870.03,  918.365,  966.7,  1015.035, 1063.37, 1111.705,
 1160.04, 1208.375, 1256.71, 1305.045, 1353.38, 1401.715, 1450.05, 1498.385,
 1546.72, 1595.055, 1643.39, 1691.725, 1740.06, 1788.395, 1836.73, 1885.065,
 1933.4,  1981.735, 2030.07, 2078.405, 2126.74, 2175.075, 2223.41, 2271.745,
 2320.08, 2368.415, 2416.75, 2465.085, 2513.42, 2561.755, 2610.09, 2658.425,
 2706.76, 2755.095, 2803.43, 2851.765, 2900.1,  2948.435, 2996.77, 3045.105,
 3093.44, 3141.775, 3190.11, 3238.445, 3286.78, 3335.115, 3383.45, 3431.785,
 3480.12, 3528.455, 3576.79, 3625.125, 3673.46, 3721.795, 3770.13, 3818.465,
 3866.8,  3915.135, 3963.47, 4011.805, 4060.14, 4108.475, 4156.81, 4205.145,
 4253.48, 4301.815, 4350.15, 4398.485, 4446.82, 4495.155, 4543.49, 4591.825,
 4640.16, 4688.495, 4736.83, 4785.165, 4833.5,  4881.835, 4930.17, 4978.505,
 5026.84, 5075.175, 5123.51, 5171.845, 5220.18, 5268.515, 5316.85, 5365.185,
 5413.52, 5461.855, 5510.19, 5558.525, 5606.86, 5655.195, 5703.53, 5751.865,
 5800.2,  5848.535, 5896.87, 5945.205, 5993.54, 6041.875, 6090.21, 6138.545,
 6186.88, 6235.215, 6283.55, 6331.885, 6380.22, 6428.555, 6476.89, 6525.225,
 6573.56, 6621.895, 6670.23, 6718.565, 6766.9,  6815.235, 6863.57, 6911.905,
 6960.24, 7008.575, 7056.91, 7105.245, 7153.58, 7201.915, 7250.25, 7298.585,
 7346.92, 7395.255, 7443.59, 7491.925, 7540.26, 7588.595, 7636.93, 7685.265,
 7733.6,  7781.935, 7830.27, 7878.605, 7926.94, 7975.275, 8023.61, 8071.945,
 8120.28, 8168.615, 8216.95, 8265.285, 8313.62, 8361.955, 8410.29, 8458.625,
 8506.96, 8555.295, 8603.63, 8651.965, 8700.3,  8748.635, 8796.97, 8845.305,
 8893.64, 8941.975, 8990.31, 9038.645, 9086.98, 9135.315, 9183.65, 9231.985,
 9280.32, 9328.655, 9376.99, 9425.325, 9473.66, 9521.995, 9570.33, 9618.665]

p_total = [25461215, 181570, 35642, 11441, 4923, 2530, 1416, 805, 508, 349, 233,
 182, 124, 91, 62, 58, 41, 37, 31, 16, 17, 22, 9, 13, 6, 11, 6, 7, 2, 7, 4, 3,
 0, 2, 1, 4, 2, 1, 1, 0, 2, 1, 1, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0]

p_Nlines = 25701407
p_low,p_high = (0, 9667)

def main():

    fakeDataset =  False
    lazy = not fakeDataset and True 

    # GO
    dataset_path = './fake_dataset/train_triplets_MSD-AG.txt' if fakeDataset else './dataset/train_triplets_MSD-AG.txt'

    nbins = 200
    Nlines = p_Nlines if lazy else countLines(dataset_path)
    print('NLines:', Nlines)
    chunkSize = 1000 if not fakeDataset else 2
    print('ChunkSize:',chunkSize)

    # get low and high for histogram boundaries
    low,high =  (p_low, p_high) if lazy else compute_boundaries( dataset_path, chunkSize, Nlines) 
    print('Plays -> Lowest:', low, ', Highest:', high )
    # read data
    bin_edges,total = ( p_bin_edges ,p_total ) if lazy else read_data( dataset_path, chunkSize, low,high,nbins, Nlines)

    N = 65

    plot_histogram(bin_edges[:N],total[:N-1])

main()