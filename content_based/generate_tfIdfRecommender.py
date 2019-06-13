"""
    generate_tfIdfRecommender reads a file organized in
    id text
    and buiilds and save a TfidfObject with its corups
"""
from ReadSave import *
import pandas as pd

from content_based.TfidfRecommender import TfidfRecommender

def generate_tfIdfRecommender(datasetPath):
    precomputed_path = datasetPath + 'precomputed_data/'
    bios_path = datasetPath + 'bios.txt'
    ds_bios = pd.read_csv(bios_path,sep='\t')
    print('Initializating TfIdfRecommender') 
    tfIdfRecommender = TfidfRecommender(ds_bios['bio'].tolist())
    save_object(tfIdfRecommender, precomputed_path + 'tfIdfRecommender.pkl')
