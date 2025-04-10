#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:30:52 2023

@author: annagalakhova
"""

from pathlib import Path
import seaborn as sns
import pandas as pd 
import numpy as np
import scipy as sp
import datetime
from ipfx.dataset.create import create_ephys_data_set
from ipfx.feature_extractor import SpikeFeatureExtractor
import matplotlib.pyplot as plt
from scipy.signal import order_filter
from scipy.signal import butter,filtfilt
import os
import sys
import importlib
sys.path.append('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/scripts')
import toolkit
from toolkit import *
from tqdm import tqdm 

#%%get major information
path = '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/data_to_analyse'
files = [f for f in os.listdir(path) if f.endswith('.nwb')]
metadata=pd.read_excel('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/metadata.xlsx')
savedir = '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/analysis/results'

#%%load in the data
data = {}
total_sweeps = {}
for name in tqdm(files, desc='Loading files', unit='file'):
    data[str(name)] = create_ephys_data_set(nwb_file=path+'/'+name) #dictionary containing all the dataset, with key = file name
    total_sweeps[str(name)] = data[str(name)].filtered_sweep_table() #dictionary containing all the raw metadta, key = file name
del name
#%%adding additional metadata into the sweep table, QC, save the newest version of swep table
file=files[0]
print(file)
dataset=data[file]
sweeps=total_sweeps[file]
sweeps=process_sweep_table(file,sweeps,metadata)
#taking out the sweeps we don't want to use
stims=['_2_5','_5','_10','_20']
if any(any(stim in element for stim in stims) for element in sweeps.stimulus_code.unique()):
    check=sweeps[sweeps.stimulus_code.str.contains('|'.join(stims))]
    for i in check.sweep_number:
        sweep = dataset.sweep(sweep_number= i)
        #filt = artefact_filtering(dataset,sweep.v, show_plot=False)
        plt.figure()
        plt.plot(sweep.v)
        #plt.plot(filt)
        plt.title(i, fontsize=13)
#leave the figures, representing sweeps you want to exclude, open, and then run the code below,
#this will write down the sweep numbers you want to exclude
n_openfigs = plt.get_fignums()
to_exclude= []
for n_fig in n_openfigs:
    fig = plt.figure(n_fig)
    title = fig.gca().get_title()
    n_sweep = int(title)
    to_exclude.append(n_sweep)

#if you need to add more to it manually
#to_exclude.append(1053)

#do the same but leave the spiking sweeps open   
n_openfigs = plt.get_fignums()
spiking= []
for n_fig in n_openfigs:
    fig = plt.figure(n_fig)
    title = fig.gca().get_title()
    n_sweep = int(title)
    spiking.append(n_sweep)


if any('hypo' in code for code in sweeps.stimulus_code.unique()):
    check=sweeps[sweeps.stimulus_code.str.contains('hypo')]
    for i in check.sweep_number:
        sweep = dataset.sweep(sweep_number= i)
        #filt = artefact_filtering(dataset,sweep.v, show_plot=False)
        plt.figure()
        plt.plot(sweep.v)
        #plt.plot(filt)
        plt.title(i, fontsize=13)

n_openfigs = plt.get_fignums()
havesag= []
for n_fig in n_openfigs:
    fig = plt.figure(n_fig)
    title = fig.gca().get_title()
    n_sweep = int(title)
    havesag.append(n_sweep)

sweeps['status'] = 'included'
# Update values based on indices using .loc
sweeps.loc[to_exclude, 'status'] = 'excluded'
sweeps.loc[spiking, 'status'] = 'spiking'
sweeps.loc[havesag, 'status'] = 'have sag'

sweeps_cleared = sweeps[sweeps.status.values != 'excluded']
del check,stims, i, sweep,to_exclude, n_openfigs, havesag, n_fig, fig, title, n_sweep, spiking,


#Rm check
rm_check=rm_check3(file,sweeps_cleared, dataset)

#check the Ra and get the readout of this
QC_ra= ra_rin_check(file, sweeps,dataset,exp=2)

#save the quality check tables
sweeps.to_csv(savedir+'/QC_files/sweeps_total_'+file[:-4]+'.csv')
sweeps_cleared.to_csv(savedir+'/QC_files/sweeps_cleared_'+file[:-4]+'.csv')
rm_check.to_csv(savedir+'/QC_files/Rm_'+file[:-4]+'.csv')
QC_ra.to_csv(savedir+'/QC_files/Ra_'+file[:-4]+'_new.csv')
#%% analyse the data
for file in files:
    ifolder=savedir+'/i_data/'+file[:-4]
    if not os.path.exists(ifolder):
        os.makedirs(ifolder)
    else:
        print(f"The directory {ifolder} already exists.")

    passives, sags =passive_sag_set2(dataset,sweeps_cleared,file,ifolder)
    peaks_total, spiking_peaks=find_peaks_set(dataset,path,file,ifolder,sweeps_cleared)
    graph, actives, spikes =  excitability_set(dataset, sweeps_cleared,ifolder, file)


    #add other data on excitability to the spiking peaks dataframe
    summary = summarise(passives, actives, peaks_total, spiking_peaks)
    #match the sweeps for sag
    

        
sns.lineplot(data=summary, x='set_number', y='rin_fit')

#%%
