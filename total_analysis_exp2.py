#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:52:51 2023

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

#%%specify paths 
path = '/Users/annagalakhova/PhD INF CNCR VU/DATA/GRM3 project/NMDA_AMPA_stimulation/analysed_files/included'
QC_path = '/Users/annagalakhova/PhD INF CNCR VU/DATA/GRM3 project/NMDA_AMPA_stimulation/results/QC_tables'
files = [f for f in os.listdir(path) if f.endswith('.nwb')]
QC_files = os.listdir(QC_path)
savedir = '/Users/annagalakhova/PhD INF CNCR VU/DATA/GRM3 project/NMDA_AMPA_stimulation/results/MODIFIED/i_tables'
#%%loading in the data and performing initial QC on ephys parameters - bulk loading
data = {}
for name in tqdm(files, desc='Loading files', unit='file'):
    data[str(name)] = create_ephys_data_set(nwb_file=path+'/'+name) #dictionary containing all the dataset, with key = file name
del name

#%%load in the data
file=files[1]
print(file)
dataset=create_ephys_data_set(nwb_file=path+'/'+file)
#dataset=data[file]
QCs = [i for i in QC_files if file[:-4] in i]
if len(QCs)>0:
    for item in QCs:
        if item.startswith('sweeps'):
            sweeps = pd.read_csv(QC_path+'/'+item)
else:
    sweeps=dataset.filtered_sweep_table()
protocols=np.unique(sweeps.stimulus_code)
#%%analyse cellular features
#create a folder where to make save the data
file=files[1]
print(file)
dataset=data[file]
QCs = [i for i in QC_files if file[:-4] in i]
if len(QCs)>0:
    for item in QCs:
        if item.startswith('sweeps'):
            sweeps = pd.read_csv(QC_path+'/'+item)
else:
    sweeps=dataset.filtered_sweep_table()
protocols=np.unique(sweeps.stimulus_code)

ifolder=savedir+'/i_figures/'+file[:-4]
if not os.path.exists(ifolder):
    os.makedirs(ifolder)
else:
    print(f"The directory {ifolder} already exists.")
#cell passive properties
if any('TestPulse' in protocol or 'rmp' in protocol for protocol in protocols):
    passivep = passive_features(file,dataset, sweeps, ifolder)
#cell_sag
if any('CCsteps' in protocol for protocol in protocols):
    sag=calculate_sag(file, dataset,sweeps,stim='CCsteps', mode=0)
    sags_df=find_matching_sweep(dataset,sag,ifolder,clamp_mode='not_clamped')
    sags_dfcl=find_matching_sweep(dataset,sag,ifolder,clamp_mode='clamped')
    #excitability
    steps = excitability(file,dataset,sweeps, ifolder)
#correct the sag
#sagsdict={}
# i=1
# j=95
# plt.figure()
# plt.plot(dataset.sweep(sweep_number=i).v)
# plt.plot(dataset.sweep(sweep_number=j).v)
# sagsdict[(file[:-4]+str(i))] = dataset.sweep(sweep_number=i).v
# sagsdict[(file[:-4]+str(j))] = dataset.sweep(sweep_number=j).v
#resonance
if any('CHIRP' in protocol for protocol in protocols):
    res=res_freq(dataset,sweeps,file,ifolder)
#ps protocols
if any('Rheo' in protocol or 'SubTh' in protocol for protocol in protocols):
    ps_sag=calculate_sag(file, dataset,sweeps,stim='SubTh',mode=0)
    sags_dfps=find_matching_sweep(dataset,ps_sag,ifolder)
    ps_rheo=calc_ps_rheo(file,dataset, sweeps, ifolder)

#if rheibase did not pass, print it to see why
# df=sweeps[sweeps.stimulus_code.str.contains('Rheo')]
# for swp in df.sweep_number:
#     plt.figure()
#     plt.plot(dataset.sweep(sweep_number=swp).v)
#     plt.plot(dataset.sweep(sweep_number=swp).i)
#     plt.title(swp)
    
#%%analyse stimulation
#peaks
avg_baseline5=get_avg(dataset,sweeps,'X06_baaseline','CurrentClamp', mode=0)
avg_naag5=get_avg(dataset,sweeps,'X18_naag','CurrentClamp',mode=0)
if any('X7_washout' in protocol for protocol in protocols):
    avg_washout5=get_avg(dataset,sweeps,'X7_washout','CurrentClamp')
    peaks5=find_peaks(dataset,avg_baseline5,avg_naag5,5,path,file, ifolder,sweeps,avg_washout5)
else:
    #peaks5=find_peaks(dataset,avg_baseline5,avg_naag5,5,path,file, ifolder,sweeps)
    peaks5,naag5sp=find_peaks(dataset,avg_baseline5,avg_naag5,5,path,file, ifolder,sweeps)
peaks5=norm_to_first(peaks5,'amplitude')
for i in peaks5.columns[5:-6].append(peaks5.columns[-3:-1]):
    peaks5=norm_to_baseline(peaks5,i)
del i

if len(sweeps[(sweeps.stimulus_code.str.contains('X3_baaseline')) & (sweeps.clamp_mode.values == 'VoltageClamp')])>0:
    avg_baseline5vc=get_avg(dataset,sweeps,'X3_baaseline','VoltageClamp')
    avg_naag5vc=get_avg(dataset,sweeps,'X5_naag','VoltageClamp')
    if any('X7_washout' in protocol for protocol in protocols):
        avg_washout5vc=get_avg(dataset,sweeps,'X7_washout','VoltageClamp')
        peaks5vc=find_peaks(dataset,avg_baseline5vc,avg_naag5vc,5,path,file, ifolder,sweeps, avg_washout5vc,cmode='VC')
    else:
        peaks5vc=find_peaks(dataset,avg_baseline5vc,avg_naag5vc,5,path,file, ifolder,sweeps, cmode='VC')
    peaks5vc=norm_to_first(peaks5vc,'amplitude')
    for i in peaks5vc.columns[5:-6].append(peaks5vc.columns[-3:-1]):
        peaks5vc=norm_to_baseline(peaks5vc,i)
    del i

if any('20' in protocol for protocol in protocols):
    avg_baseline20=get_avg(dataset,sweeps,'resp20_ba','CurrentClamp', mode=0)
    avg_naag20=get_avg(dataset,sweeps,'resp20_na','CurrentClamp',mode=0)
    if any('resp20_washout' in protocol for protocol in protocols):
        avg_washout20=get_avg(dataset,sweeps,'resp20_wa','CurrentClamp')
        peaks20=find_peaks(dataset,avg_baseline20,avg_naag20,20,path,file, ifolder,sweeps,avg_washout20)
    else:
        peaks20=find_peaks(dataset,avg_baseline20,avg_naag20,20,path,file, ifolder,sweeps)
    peaks20=norm_to_first(peaks20,'amplitude')
    for i in peaks20.columns[5:7]:
        peaks20=norm_to_baseline(peaks20,i)
    del i
    
for i in range(222,242):
    plt.figure()
    plt.plot(dataset.sweep(sweep_number=i).v)
    plt.title(i)


if any('100' in protocol for protocol in protocols):
    avg_baseline100=get_avg(dataset,sweeps,'baseline_100','CurrentClamp', mode=0)
    avg_naag100=get_avg(dataset,sweeps,'naag_100','CurrentClamp',mode=0)
    if any('X8_washout_100' in protocol for protocol in protocols): #check out the name of washout protocol
        avg_washout100=get_avg(dataset,sweeps,'X8_washout_100','CurrentClamp')
        peaks100=find_peaks100(avg_baseline100,avg_naag100,path,file, ifolder,sweeps,avg_washout100)
    else:
        peaks100=find_peaks100(avg_baseline100,avg_naag100,path,file, ifolder,sweeps)
    peaks100=norm_to_first(peaks100,'amplitude')
    peaks100=norm_to_baseline(peaks100,'amplitude')


#%%combine all the data into one table
#saving all the peaks together
if 'peaks5' in locals():
    all_peaks = peaks5
# Check if peaks5vc exists and concatenate it
if 'peaks5vc' in locals():
    all_peaks = pd.concat([all_peaks, peaks5vc])
# Check if peaks20 exists and concatenate it
if 'peaks20' in locals():
    all_peaks = pd.concat([all_peaks, peaks20])
# Check if peaks100 exists and concatenate it
if 'peaks100' in locals():
    all_peaks = pd.concat([all_peaks, peaks100])
if 'allpeaks' in locals():   
    all_peaks.to_csv(savedir+'/allpeaks_'+file[:-4]+'.csv')
if 'naag5sp' in locals():  
    naag5sp.to_csv(savedir+'/NAAGspikes_'+file[:-4]+'.csv')
#saving all sagd together
if 'sags_dfps' in globals():
    total_sag=pd.concat([sags_df,sags_dfps])
if 'sags_dfcl' in globals():  
    total_sag=pd.concat([sags_df,sags_dfcl])
else:
    total_sag=sags_df
total_sag.to_csv(savedir+'/isags_'+file[:-4]+'.csv')
if 'sag_ps' in globals():
    sags=pd.concat([sag,ps_sag])
else:
    sags=sag
sags.to_csv(savedir+'/allsags_'+file[:-4]+'.csv')
#save excitability graphs
if 'steps' in locals():
    for key in steps:
        temp=steps[key]
        if len(temp) !=0:
            temp.to_csv(savedir+'/'+key+file[:-4]+'.csv')
    del temp
#save the rest
rest=passivep
if 'ps_rheo' in locals() and ps_rheo is not None:
    rest=passivep.merge(ps_rheo, on=['cell', 'condition'], how='outer')
if 'res' in locals():
    rest=passivep.merge(res, on=['cell', 'condition'], how='outer')
rest.to_csv(savedir+'/passive_'+file[:-4]+'.csv')

del avg_baseline5, avg_naag5,peaks5,all_peaks 
del sags_df,total_sag,sag,sags,
del sags_dfcl
del steps, passivep, rest, key
del peaks20, avg_baseline20, avg_naag20
del avg_washout5, avg_washout5vc, avg_washout20
del ps_rheo, ps_sag, sags_dfps,
del res
del avg_baseline5vc, avg_naag5vc,avg_baseline20, avg_naag20
del peaks100, avg_baseline100, avg_naag100
del protocols, ifolder, sweeps
del peaks5vc

#%% get all the data together
#data to referr to from the grouping
metadata = pd.read_excel('/Users/annagalakhova/PhD INF CNCR VU/DATA/GRM3 project/NMDA_AMPA_stimulation/new_data/metadata.xlsx')
metadata = metadata[metadata['Ra_start'].notnull()]
metadata['Group'] = metadata['Group'].astype(str)
data_dict = metadata.set_index('cell')['Group'].to_dict()
#%%getting total
peaks_total=process_and_save_data(savedir, 'allpeaks', data_dict)
sags_total=process_and_save_data(savedir, 'allsags', data_dict)
isags_total=process_and_save_data(savedir, 'isags', data_dict)
grs_total=process_and_save_data(savedir, 'graph_not_', data_dict)
pas_total=process_and_save_data(savedir, 'passive', data_dict)
spks_total=process_and_save_data(savedir, 'spikes_not', data_dict)
grsc_total=process_and_save_data(savedir, 'graph_', data_dict)
spksc_total=process_and_save_data(savedir, 'spikes_', data_dict)
active_total=get_active(savedir,grs_total)
active_total=get_active(savedir,grsс_total, mode='cl')

#%%to get a certain protocol for all the cells
for file in tqdm(files, desc='Loading files', unit='file'):
    dataset = create_ephys_data_set(nwb_file=path+'/'+file)
    sweeps = dataset.filtered_sweep_table()
    protocols=np.unique(sweeps.stimulus_code)
    ifolder=savedir[:-9]+'/i_figures/'+file[:-4]
    if any('CCsteps' in protocol for protocol in protocols):
        sag=calculate_sag(file, dataset,sweeps,stim='CCsteps', mode=1)
        matching_sags,sags_df=find_matching_sweep(dataset,sag,ifolder,clamp_mode='not_clamped')
    if 'sag' in locals() and sag is not None:
        sag.to_csv(savedir+'/nsag_'+file[:-4]+'.csv')
    if 'sags_df' in locals() and sags_df is not None:
        sags_df.to_csv(savedir+'/misag_'+file[:-4]+'.csv')


