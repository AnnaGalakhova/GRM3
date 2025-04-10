#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:20:13 2024

@author: annagalakhova
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from ipfx.dataset.create import create_ephys_data_set
import h5py
import toolkit
from toolkit import *
import logging
from datetime import datetime, timedelta
current_date=datetime.now().date()

#%% constants - directories, subdirectories and general information
path = '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project'
# List all .nwb files in the directory
all_files = [f for f in os.listdir(os.path.join(path, 'all_data')) if f.endswith('.nwb')]
try:
    overview = pd.read_excel(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), sheet_name=f'analysed_{current_date}')
except:
    overview = pd.read_excel(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), sheet_name=-1)
#%% functions
#load in the data and metadata, perfprm Ra analysis
def monoExp(x, m, k, b):
    return m * np.exp(-k * x) + b

def find_correct_file(cell, all_files):
    """
    Finds the correct file from a list of files based on a given 'cell' name.
    
    Arguments:
    cell -- The target cell string (e.g., 'H24.29.268.11.61.01')
    all_files -- List of filenames to search through
    
    Returns:
    correct_file -- The filename that matches the 'cell' pattern best, or None if no match is found
    """
    # First, look for an exact match (including .nwb)
    exact_matches = [f for f in all_files if f.startswith(cell)]

    if exact_matches:
        return exact_matches[0]  # Return the first exact match

    # If no exact match, use partial matching as a fallback
    partial_matches = [f for f in all_files if cell[:17] in f and cell[17:] in f[17:-4]]  # Match without .nwb
    if partial_matches:
        return partial_matches[0]  # Return the first partial match

    return None 

def load_data_metadata(path, s, ref=0):
    rectype = None
    sweeps = None
    dataset = None
    #load in the metadata
    try:
        dataset = create_ephys_data_set(nwb_file=path+'/all_data/'+s)
        try:
            sweeps=dataset.filtered_sweep_table() 
            rectype = 'mono'
            sweeps['cell']= s[:-4]
            #perform Ra check
            for index,row in sweeps.iterrows():
                if row.clamp_mode == "VoltageClamp":
                    with h5py.File(f'{path}/all_data/{s}', 'r') as f:
                        nzeros=5-len(str(row.sweep_number))
                        indexx = '0' * nzeros + str(row.sweep_number)
                        hs_nr=list(f['acquisition'].keys())[0][-1]
                        if hs_nr !=0:
                            rectype = 'mono_channelnonzero'
                        if 'whole_cell_capacitance_comp' in f['acquisition']['data_'+str(indexx)+'_AD'+str(hs_nr)].keys(): #think of how to implement PR AD1 for the future if I figure out channels in ipfx
                            sweeps.at[index, 'Ra'] = f['acquisition']['data_000'+str(indexx)+'_AD'+str(hs_nr)]['whole_cell_series_resistance_comp'][0]*0.000001
                            sweeps.at[index, 'compensation'] = 'yes'
                        else:
                            if rectype == 'mono':
                                sweep = dataset.sweep(sweep_number=index)
                                if len(sweep.v) == sweep.epochs['sweep'][1]+1:
                                    tp_start, tp_end = sweep.epochs['test'][0],sweep.epochs['test'][1]
                                    v=sweep.v[tp_start: tp_end]
                                    i=sweep.i[tp_start: tp_end]
                                    if len(v) > 0:
                                        stim_index = np.nonzero(v)[0][0]
                                        bsl = i[stim_index-100]
                                        peak = max(i)
                                        tp_inp= round(max(v))
                                        ra=(tp_inp/(peak-bsl))*1000
                                        sweeps.at[index, 'Ra'] = ra
                                        sweeps.at[index, 'compensation'] = 'no'
                                        sweeps.at[index, 'compensation'] = 'no'
                                    else:
                                        continue
                            elif rectype == 'mono_channelnonzero':
                                with h5py.File(f'{path}/all_data/{s}', 'r') as f:
                                    sweepi = f['acquisition']['data_'+str(indexx)+'_AD'+str(hs_nr)]['data']
                                    sweepv=f['stimulus']['presentation']['data_'+str(indexx)+'_DA'+str(hs_nr)]['data']
                                    fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
                                    tp_start, tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[0], ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
                                    v=sweepv[tp_start: tp_end]
                                    i=sweepi[tp_start: tp_end]
                                    if len(v) > 0:
                                        stim_index = np.nonzero(v)[0][0]
                                        bsl = i[stim_index-100]
                                        peak = max(i)
                                        tp_inp= round(max(v))
                                        ra=(tp_inp/(peak-bsl))*1000
                                        sweeps.at[index, 'Ra'] = ra
                                        sweeps.at[index, 'compensation'] = 'no'
                                    else:
                                        continue  # Skip if 'v' is empty
                     
            ra_norm_factor = sweeps['Ra'].dropna().loc[ref] #choose here you first reference Ra
            # Normalize 'Ra' by the Ra_norm_factor, ensuring NaN stays as NaN
            sweeps['Ra_norm'] = sweeps['Ra'] / ra_norm_factor
            # Handle cases where Ra is NaN to ensure Ra_norm also remains NaN
            sweeps['Ra_norm'] = sweeps.apply(lambda row: np.nan if np.isnan(row['Ra']) else row['Ra_norm'], axis=1)
                
        except Exception as e:
            print(f'Potentially recording has multiple channels : {e}')
        # If the error is related to "tuple", switch to using h5py
        if "tuple" in str(e) and sweeps is None:
            rectype = 'multi'
            try:
                with h5py.File(f'{path}/all_data/{s}', 'r') as f:
                    #get the sweeps data into the table
                    sweeps=pd.DataFrame(list(f['acquisition'].keys()))
                    sweeps['sweep_number'] = sweeps[0].apply(lambda x: int(x.split('_')[1]))
                    hs_nr=sweeps[0].apply(lambda x: 'HS'+x.split('_')[2][-1])
                    sweeps['channel'] = hs_nr
                    sweeps = sweeps.rename(columns={0: 'key_name'})
                    #construct a sweep table
                    for i in range(0,len(f['general']['labnotebook']['ITC18_Dev_0']['textualValues'])):
                        temp=f['general']['labnotebook']['ITC18_Dev_0']['textualValues'][i]
                        sweep_number=int(temp[0][0])
                        stimulus_code=next((f for f in temp[5] if '_DA_0'  in f), None)
                        if stimulus_code is None:
                            stimulus_code = 'no data'
                        unit=next((f for f in temp[7] if 'mV' in f or 'pA' in f), None)
                        if unit is None:
                            unit = 'no data'
                        if 'mV' in unit:
                            stimulus_unit='Amps'
                            clamp_mode='CurrentClamp'
                        elif 'pA' in unit:
                            stimulus_unit='Volts'
                            clamp_mode='VoltageClamp'
                        sweeps.at[sweeps['sweep_number'] == sweep_number, 'stimulus_code'] = stimulus_code[:-5]
                        sweeps.at[sweeps['sweep_number'] == sweep_number, 'stimulus_unit'] = stimulus_unit
                        sweeps.at[sweeps['sweep_number'] == sweep_number, 'clamp_mode'] = clamp_mode
                    #add metadata to sweep table
                    if len(sweeps.channel.unique())>1 and s.startswith('H'):
                        sweeps.at[sweeps['channel'] == 'HS0', 'cell'] = s[:-6]
                        sweeps.at[sweeps['channel'] == 'HS1', 'cell'] = s[:-8]+s[-6:-4]
                    elif len(sweeps.channel.unique())>1 and s.startswith('M'):
                        sweeps.at[sweeps['channel'] == 'HS0', 'cell'] = s[:-4]
                        sweeps.at[sweeps['channel'] == 'HS1', 'cell'] = s[:-6]+s[-4:-2]
                #perform Ra check
                for index,row in sweeps.iterrows():
                   if row.clamp_mode == "VoltageClamp":
                        with h5py.File(f'{path}/all_data/{s}', 'r') as f:
                            if 'whole_cell_capacitance_comp' in f['acquisition'][row.key_name].keys(): 
                                sweeps.at[sweeps.key_name == row.key_name, 'Ra'] = f['acquisition'][row.key_name]['whole_cell_series_resistance_comp'][0]*0.000001
                                sweeps.at[sweeps.key_name == row.key_name, 'compensation'] = 'yes'
                                sweeps.at[sweeps.key_name == row.key_name, 'leak_pa'] = f['acquisition'][row.key_name]['data'][150]
                            else:
                                sweepi=f['acquisition'][row.key_name]['data']
                                sweepv=f['stimulus']['presentation'][row.key_name[:-4]+'_DA'+row.key_name[-1]]['data']
                                tp_start, tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[0], ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
                                v=sweepv[: tp_end]
                                i=sweepi[: tp_end]
                                stim_index = np.nonzero(v)[0][0]
                                bsl = i[stim_index-100]
                                peak = max(i)
                                tp_inp= round(max(v))
                                ra=(tp_inp/(peak-bsl))*1000
                                if ra == np.inf:
                                    ra=np.nan
                                sweeps.at[sweeps.key_name == row.key_name, 'Ra'] = ra
                                sweeps.at[sweeps.key_name == row.key_name, 'compensation'] = 'no'
                                sweeps.at[sweeps.key_name == row.key_name, 'leak_pa'] = bsl
                    
                sweeps['Ra_norm_factor'] = sweeps.groupby('channel')['Ra'].transform(lambda x: x.dropna().iloc[ref] if not x.dropna().empty else np.nan)
                # Normalize 'Ra' by the first non-NaN value per channel, ensuring NaN stays as NaN
                sweeps['Ra_norm'] = sweeps['Ra'] / sweeps['Ra_norm_factor']
                # Handle cases where Ra is NaN to ensure Ra_norm also remains NaN
                sweeps['Ra_norm'] = sweeps.apply(lambda row: np.nan if np.isnan(row['Ra']) else row['Ra_norm'], axis=1)
                sweeps=sweeps.drop(columns=['Ra_norm_factor'])
            except Exception as e:
                rectype = 'failed'
                sweeps=pd.DataFrame()
                print("An error occurred with h5py:", e)
        else: 
            print(f"An error occurred with filtered_sweep_table(): {e}")
            rectype = 'failed'
            sweeps=pd.DataFrame()
    except Exception as e:          
        print(f'an error occured loading the metadata {e}')
        return dataset, rectype, sweeps, hs_nr

#assign conditions taken from the overview-derived df
def assign_condition_based_on_index(df, sweeps):
    conditions = []
    unique_conditions= df['condition'].unique()
    for i, condition in enumerate(unique_conditions):
        # Get the start index for the current condition
        start_index = df[df['condition'] == condition]['fswp_condition'].iloc[0]
        if condition != unique_conditions[-1]:
            end_index = df[df['condition'] == unique_conditions[i+1]]['fswp_condition'].iloc[0] -1
        else:
            end_index = len(sweeps)-1
    
    
        # Append the condition and its index range to the list
        conditions.append((start_index, end_index, condition))
    sweeps['condition'] = 'unknown'  # Initialize all conditions as 'unknown'
    # Iterate over each condition range in the `conditions` list
    for start_index, end_index, condition in conditions:
        # Assign the condition to the corresponding rows in `sweeps`
        sweeps.loc[start_index:end_index, 'condition'] = condition
    sweeps['QC'] = np.nan
    sweeps.loc[(sweeps['Ra_norm'] >= 0.7) & (sweeps['Ra_norm'] <= 1.3), 'QC'] = 'included'
    sweeps.loc[(sweeps['Ra_norm'] < 0.7) | (sweeps['Ra_norm'] > 1.3), 'QC'] = 'excluded'
    sweeps['QC'].fillna(method='ffill', inplace=True)
    return sweeps

def load_data(path, s):
    rectype = None
    dataset = None
    #load in the metadata
    try:
        dataset = create_ephys_data_set(nwb_file=path+'/all_data/'+s)
        rectype = 'mono'
        sweeps=dataset.filtered_sweep_table() 
    except Exception as e:
        print(f'Potentially recording has multiple channels : {e}')
        # If the error is related to "tuple", switch to using h5py
        if "tuple" in str(e):
            rectype = 'multi'
        else: 
            rectype = 'failed'
    return dataset, rectype

def process_sweep_table(rectype,df, dataset,s,cell, ref=0, beg_swp=0):
    sweeps=None
    hs_nr = None
    
    if rectype == 'mono':
        sweeps=dataset.filtered_sweep_table() 
        sweeps['cell']= s[:-4]
        sweeps=sweeps[beg_swp:]
        f=h5py.File(f'{path}/all_data/{s}', 'r')
        hs_nr=int(list(f['acquisition'].keys())[beg_swp][-1])
        for index,row in sweeps.iterrows():
            if row.clamp_mode == "VoltageClamp":
                nzeros=5-len(str(row.sweep_number))
                indexx = '0' * nzeros + str(row.sweep_number)
                unique_channels = set(name.split('_')[-1] for name in f['acquisition'].keys())
                if len(unique_channels) > 1:
                    hs_nr=int(list(f['acquisition'].keys())[-1][-1])
                hs_nr=int(list(f['acquisition'].keys())[beg_swp][-1])
                if 'whole_cell_capacitance_comp' in f['acquisition']['data_'+str(indexx)+'_AD'+str(hs_nr)].keys(): 
                    print(hs_nr)
                    sweeps.at[index, 'Ra'] = f['acquisition']['data_'+str(indexx)+'_AD'+str(hs_nr)]['whole_cell_series_resistance_comp'][0]*0.000001
                    sweeps.at[index, 'compensation'] = 'yes'
                else:
                    print(f'cound not find the data for the sweep {indexx}')
                    sweeps.at[index, 'compensation'] = 'no'
                    try:
                        sweep = dataset.sweep(sweep_number=row.sweep_number)
                        tp_start, tp_end = ((np.where(sweep.v[:-1] != sweep.v[1:])[0]) + 1)[0], ((np.where(sweep.v[:-1] != sweep.v[1:])[0]) + 1)[1]
                        if len(sweep.v) > tp_end:
                            v=sweep.v[tp_start-100: tp_end]
                            i=sweep.i[tp_start-100: tp_end]
                            if v[-1] < v[0]:
                                v=-v
                                i=-i
                            use_h5py = False
                        else: 
                            print(f'incomplete sweep {row.sweep_number}')
                            ra = np.nan
                            sweeps.at[index, 'Ra'] = ra
                            use_h5py = False
                            
                    except Exception as e:
                         print(f'Could not open the data with h5py for TP correction : {e}')
                         use_h5py = True
                         
                    if use_h5py:     
                        sweepi = f['acquisition']['data_'+str(indexx)+'_AD'+str(hs_nr)]['data']
                        sweepv=f['stimulus']['presentation']['data_'+str(indexx)+'_DA'+str(hs_nr)]['data']
                        fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
                        tp_start, tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[0], ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
                        v=sweepv[tp_start-100: tp_end]
                        i=sweepi[tp_start-100: tp_end]
                    
                    #get the values to get Rin, leak and tau
                    try:
                        bsl = np.mean(i[0:100])
                        peak = max(i)
                        tp_inp= round(max(v))
                        ra=(tp_inp/(peak-bsl))*1000
                        sweeps.at[index, 'Ra'] = ra
                    except Exception as e:
                        print(f'data not calculatable for sweep {row.sweep_number}')
                        ra=np.nan
                        sweeps.at[index, 'Ra'] = ra
            else: 
                    print(f'no VC sweeps')
                    ra = np.nan
                    sweeps.at[index, 'Ra'] = ra
        if len(sweeps[sweeps.clamp_mode == 'VoltageClamp']) != 0:
            ra_norm_factor = sweeps['Ra'].dropna().iloc[ref] #choose here you first reference Ra
            # Normalize 'Ra' by the Ra_norm_factor, ensuring NaN stays as NaN
            sweeps['Ra_norm'] = sweeps['Ra'] / ra_norm_factor
            # Handle cases where Ra is NaN to ensure Ra_norm also remains NaN
            sweeps['Ra_norm'] = sweeps.apply(lambda row: np.nan if np.isnan(row['Ra']) else row['Ra_norm'], axis=1)  
        else:
            ra = np.nan
            sweeps.at[index, 'Ra'] = ra  
            sweeps.at[index, 'Ra_norm'] = np.nan

    
    elif rectype == 'multi':
        try:
            f=h5py.File(f'{path}/all_data/{s}', 'r')
            #get the sweeps data into the table
            sweeps=pd.DataFrame(list(f['acquisition'].keys()))
            sweeps['sweep_number'] = sweeps[0].apply(lambda x: int(x.split('_')[1]))
            hs_nr=sweeps[0].apply(lambda x: 'HS'+x.split('_')[2][-1])
            sweeps['channel'] = hs_nr
            sweeps = sweeps.rename(columns={0: 'key_name'})
            #construct a sweep table
            for i in range(0,len(f['general']['labnotebook']['ITC18_Dev_0']['textualValues'])):
                temp=f['general']['labnotebook']['ITC18_Dev_0']['textualValues'][i]
                sweep_number=int(temp[0][0])
                stimulus_code=next((f for f in temp[5] if '_DA_0'  in f), None)
                if stimulus_code is None:
                    stimulus_code = 'no data'
                unit=next((f for f in temp[7] if 'mV' in f or 'pA' in f), None)
                if unit is None:
                    unit = 'no data'
                if 'mV' in unit:
                    stimulus_unit='Amps'
                    clamp_mode='CurrentClamp'
                elif 'pA' in unit:
                    stimulus_unit='Volts'
                    clamp_mode='VoltageClamp'
                
                sweeps.at[sweeps['sweep_number'] == sweep_number, 'stimulus_code'] = stimulus_code[:-5]
                sweeps.at[sweeps['sweep_number'] == sweep_number, 'stimulus_unit'] = stimulus_unit
                sweeps.at[sweeps['sweep_number'] == sweep_number, 'clamp_mode'] = clamp_mode
                
                
            #add metadata to sweep table
            if len(sweeps.channel.unique())>1 and s.startswith('H'):
                sweeps.at[sweeps['channel'] == 'HS0', 'cell'] = s[:-6]
                sweeps.at[sweeps['channel'] == 'HS1', 'cell'] = s[:-8]+s[-6:-4]
            elif len(sweeps.channel.unique())>1 and s.startswith('M'):
                sweeps.at[sweeps['channel'] == 'HS0', 'cell'] = s[:-6]
                sweeps.at[sweeps['channel'] == 'HS1', 'cell'] = s[:-8]+s[-6:-4]
            
            sweeps['holding']=np.nan
            for key_name in sweeps.key_name.unique():
                comments = f['acquisition'][key_name].attrs['comments']
                info = comments.split('\n')
                clamp_mode = sweeps[sweeps.key_name == key_name].clamp_mode.values[0] 
                
                if clamp_mode == 'CurrentClamp':
                    for line in info:
                        if 'I-Clamp Holding Enable' in line:
                            if line[3] == key_name[-1]:
                                splitline = line.split(':')
                                holding = splitline[-1]
                                sweeps.at[sweeps['key_name'] == key_name, 'holding'] = holding
                elif clamp_mode == 'VoltageClamp':
                    for line in info:
                        if 'V-Clamp Holding Enable' in line:
                            if line[3] == key_name[-1]:
                                splitline = line.split(':')
                                holding = splitline[-1]
                                sweeps.at[sweeps['key_name'] == key_name, 'holding'] = holding
            

        except Exception as e:
            hs_nr = None
            sweeps=pd.DataFrame()
            print("An error occurred with h5py:", e)
            
        try:
            #perform Ra check
            for index,row in sweeps.iterrows():
                if row.clamp_mode == "VoltageClamp":
                    if 'whole_cell_capacitance_comp' in f['acquisition'][row.key_name].keys(): 
                        sweeps.at[sweeps.key_name == row.key_name, 'Ra'] = f['acquisition'][row.key_name]['whole_cell_series_resistance_comp'][0]*0.000001
                        sweeps.at[sweeps.key_name == row.key_name, 'compensation'] = 'yes'
                        sweeps.at[sweeps.key_name == row.key_name, 'leak_pa'] = f['acquisition'][row.key_name]['data'][150]
                    else:
                        sweepi=f['acquisition'][row.key_name]['data']
                        sweepv=f['stimulus']['presentation'][row.key_name[:-4]+'_DA'+row.key_name[-1]]['data']
                        tp_start, tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[0], ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
                        v=sweepv[tp_start-100: tp_end]
                        i=sweepi[tp_start-100: tp_end]
                        
                        if v[-1] < v[0]:
                                v=-v
                                i=-i
                                
                        bsl = np.mean(i[0:100])
                        peak = max(i)
                        tp_inp= round(max(v))
                        ra=(tp_inp/(peak-bsl))*1000
                        sweeps.at[index, 'Ra'] = ra
                        if ra == np.inf:
                            ra=np.nan
                        else: 
                            continue
                        sweeps.at[sweeps.key_name == row.key_name, 'Ra'] = ra
                        sweeps.at[sweeps.key_name == row.key_name, 'compensation'] = 'no'
                        sweeps.at[sweeps.key_name == row.key_name, 'leak_pa'] = bsl
                else: 
                    print('no VC sweeps')
                    ra = np.nan
                    sweeps.at[index, 'Ra'] = ra
                    
            if len(sweeps[sweeps.clamp_mode == 'VoltageClamp']) != 0:        
                sweeps['Ra_norm_factor'] = sweeps.groupby('channel')['Ra'].transform(lambda x: x.dropna().iloc[ref] if not x.dropna().empty else np.nan)
                # Normalize 'Ra' by the first non-NaN value per channel, ensuring NaN stays as NaN
                sweeps['Ra_norm'] = sweeps['Ra'] / sweeps['Ra_norm_factor']
                # Handle cases where Ra is NaN to ensure Ra_norm also remains NaN
                sweeps['Ra_norm'] = sweeps.apply(lambda row: np.nan if np.isnan(row['Ra']) else row['Ra_norm'], axis=1)
                sweeps=sweeps.drop(columns=['Ra_norm_factor'])
            else:
                ra = np.nan
                sweeps.at[index, 'Ra'] = ra
                sweeps.at[index, 'Ra_norm'] = np.nan
                    
        
            
        except Exception as e:
            print("An error occurred with Ra performance:", e)
        
        cell_other = [x for x in sweeps.cell.unique()  if x != cell][0]
        sweeps_other = sweeps[sweeps['cell'] == cell_other]
        sweeps = sweeps[sweeps['cell'] == cell].reset_index(drop=True)
        hs_nr =  int(sweeps.key_name.iloc[0][-1])
    
    
    
    conditions = []
    unique_conditions= df['condition'].unique()
    for i, condition in enumerate(unique_conditions):
        # Get the start index for the current condition
        start_index = df[df['condition'] == condition]['fswp_condition'].iloc[0]
        if condition != unique_conditions[-1]:
            end_index = df[df['condition'] == unique_conditions[i+1]]['fswp_condition'].iloc[0] -1
        else:
            end_index = len(sweeps)-1

        # Append the condition and its index range to the list
        conditions.append((start_index, end_index, condition))
    
    init = 'unknown'
    sweeps['condition'] = init # Initialize all conditions as 'unknown'
    # Iterate over each condition range in the `conditions` list
    for start_index, end_index, condition in conditions:
        # Assign the condition to the corresponding rows in `sweeps`
        sweeps.loc[start_index:end_index, 'condition'] = condition
    qckech = np.nan
    sweeps['QC'] = qckech
    sweeps.loc[(sweeps['Ra_norm'] >= 0.7) & (sweeps['Ra_norm'] <= 1.3), 'QC'] = 'included'
    sweeps.loc[(sweeps['Ra_norm'] < 0.7) | (sweeps['Ra_norm'] > 1.3), 'QC'] = 'excluded'
    sweeps['QC'].fillna(method='ffill', inplace=True)
    sweeps['QC'].fillna(method='bfill', inplace=True)
    return sweeps, hs_nr


def TP_VC_analysis(sweeps,dataset, rectype, hs):
    results={}
    df2 = sweeps[sweeps.clamp_mode.values == 'VoltageClamp'].reset_index(drop=True)
    if df2.shape[0] != 0:
        for number, r in enumerate(df2.sweep_number):
            try:
                if rectype == 'mono':
                    try:
                        sweep = dataset.sweep(sweep_number=r)
                        if len(sweep.v) >= sweep.epochs['test'][1]+1:
                            fs=sweep.sampling_rate
                            #find where the test epoch is and define therefore the signal
                            tp_start, tp_end = ((np.where(sweep.v[:-1] != sweep.v[1:])[0]) + 1)[0], ((np.where(sweep.v[:-1] != sweep.v[1:])[0]) + 1)[1]
                            v=sweep.v[tp_start-100: tp_end]
                            i=sweep.i[tp_start-100: tp_end]
                            if v[-1] < v[0]:
                                    v=-v
                                    i=-i
                            use_h5py = False
                        else: 
                            print(f'incomplete sweep {r}')
                            use_h5py = True
                            
                            
                    except Exception as e:
                         print(f'Could not open the data with h5py for TP correction : {e}')
                         use_h5py = True
                         
                    if use_h5py:
                        f=h5py.File(f'{path}/all_data/{s}', 'r')
                        sweepi = f['acquisition']['data_'+str(indexx)+'_AD'+str(hs)]['data']
                        sweepv=f['stimulus']['presentation']['data_'+str(indexx)+'_DA'+str(hs)]['data']
                        fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
                        tp_start, tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[0], ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
                        v=sweepv[tp_start-100: tp_end]
                        i=sweepi[tp_start-100: tp_end]
                        if v[-1] < v[0]:
                                    v=-v
                                    i=-i
                        
                elif rectype == 'multi': 
                    f=h5py.File(f'{path}/all_data/{s}', 'r')
                    nzeros=5-len(str(r))
                    indexx = '0' * nzeros + str(r)
                    sweepi = f['acquisition']['data_'+str(indexx)+'_AD'+str(hs)]['data']
                    sweepv=f['stimulus']['presentation']['data_'+str(indexx)+'_DA'+str(hs)]['data']
                    fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
                    tp_start, tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[0], ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
                    v=sweepv[tp_start-100: tp_end]
                    i=sweepi[tp_start-100: tp_end]
                    if v[-1] < v[0]:
                                    v=-v
                                    i=-i
                    
                #get the values to get Rin, leak and tau
                peak = max(i)
                tau_start = np.argmax(i)
                tp_inp= round(max(v))
                bsl = np.mean(i[0:100])
                
                y=i[tau_start:]
                tp_ss=np.mean(y[-50:-1])
                rin = (tp_inp/(tp_ss-bsl))*1000
       
                #compute tau in ms
                x = np.arange(0, len(y))
                m_guess = peak-bsl
                b_guess = tp_ss
                t_guess = 0.03
                try:
                    (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess])
                    tau = 1 / (t_true) / fs * 1000
                except Exception as e:
                    print(f"Fit error: {e}")
                    tau = np.nan  # Assign NaN if the fit fails
    
                sweeps.loc[sweeps.sweep_number == r, ['rin', 'tau', 'leak']] = [rin, tau, bsl]
            except Exception as e:
                    print(f"sweep: {r} failed, moved onto another one")
                    rin = np.nan
                    tau = np.nan  # Assign NaN if the fit fails
                    bsl = np.nan
                    sweeps.loc[sweeps.sweep_number == r, ['rin', 'tau', 'leak']] = [rin, tau, bsl]
                    
        df_grp = sweeps.groupby(['condition', 'QC']).mean().reset_index()
        for cond in df_grp.condition.unique():
            results[cond] = {
                'rin': np.nan,
                'tau': np.nan,
                'leak': np.nan}
        for cond in df_grp.condition.unique():
            results[cond] = {
                'rin': df_grp[df_grp.condition == cond]['rin'].iloc[0],
                'tau': df_grp[df_grp.condition == cond]['tau'].iloc[0],
                'leak': df_grp[df_grp.condition == cond]['leak'].iloc[0]}
    else: 
        for cond in sweeps.condition.unique():
           results[cond] = {
               'rin': np.nan,
               'tau': np.nan,
               'leak': np.nan}
        print(f'no VC sweeps')
                 
    return results, sweeps

def get_rmp(sweeps,dataset, rectype):
    results={}
    if rectype == 'mono':
        df2=sweeps[(sweeps.clamp_mode == 'CurrentClamp')&(sweeps.leak_pa.isnull())]
        if df2.shape[0] != 0:
            rmp_mask = df2['stimulus_code'].str.contains('rmp', na=False)
            # Filter df2 for rows where the mask is True
            rmp_rows = df2[rmp_mask]
            if len(rmp_rows) > 0:
            # Iterate over these rows using index and sweep_number (which is 'r' in your loop)
                for index, row in rmp_rows.iterrows():
                    sweep_number = row['sweep_number']
                    swp = dataset.sweep(sweep_number=sweep_number)
                    rmp = np.mean(swp.v[swp.epochs['test'][1]:], axis=0)
                    sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'RMP'] = rmp
            
            else:
                    rmp=np.nan
                    sweeps['RMP'] = rmp
                    
            steps = df2[df2['clamp_mode'].str.contains('Current', na=False)]
            if len(steps) > 0:
                #get indexes
                sweep=dataset.sweep(sweep_number=steps.iloc[0]['sweep_number'])
                beg,end =0, ((np.where(sweep.i[:-1] != sweep.i[1:])[0]) + 1)[0]
                for index, row in steps.iterrows():
                    sweep_number = row['sweep_number']
                    swp = dataset.sweep(sweep_number=sweep_number)
                    rmp = np.mean(swp.v[beg:end], axis=0)
                    sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'RMP'] = rmp
            else:
                    rmp=np.nan
                    sweeps['RMP'] = rmp 
        else:
            rmp=np.nan
            sweeps['RMP'] = rmp
    elif rectype == 'multi':
        df2=sweeps[(sweeps.clamp_mode == 'CurrentClamp')]
        if df2.shape[0] != 0:
            f=h5py.File(f'{path}/all_data/{s}', 'r')
            rmp_mask = df2['stimulus_code'].str.contains('rmp', na=False)
            # Filter df2 for rows where the mask is True
            rmp_rows = df2[rmp_mask]
            if len(rmp_rows) > 0:
                sweepi=f['stimulus']['presentation'][rmp_rows['key_name'].iloc[0][:-3]+'DA'+rmp_rows['key_name'].iloc[0][-1]]['data']
                beg= ((np.where(sweepi[:-1] != sweepi[1:])[0]) + 1)[1]
                # Iterate over these rows using index and sweep_number (which is 'r' in your loop)
                for index, row in rmp_rows.iterrows():
                    #sweepi=f['stimulus']['presentation'][row.key_name[:-3]+'DA'+row.key_name[-1]]['data']
                    sweepv=f['acquisition'][row.key_name]['data'][beg:]
                    rmp = np.nanmean(sweepv, axis=0)
                    sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'RMP'] = rmp
            else:
                    rmp=np.nan
                    sweeps['RMP'] = rmp
                    
            steps = df2[df2['stimulus_code'].str.contains('teps', na=False)]
            if len(steps) > 0:
                if steps['holding'].iloc[0] == ' Off':
                    #get indexes
                    sweepi=f['stimulus']['presentation'][steps['key_name'].iloc[0][:-3]+'DA'+steps['key_name'].iloc[0][-1]]['data']
                    beg,end = beg,end =0, ((np.where(sweepi[:-1] != sweepi[1:])[0]) + 1)[0]
                    for index, row in steps.iterrows():
                        sweepv=f['acquisition'][row.key_name]['data'][beg+400:end]
                        rmp = np.nanmean(sweepv, axis=0)
                        sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'RMP'] = rmp
                else:
                    rmp=np.nan
                    sweeps['RMP'] = rmp
        else:
            rmp=np.nan
            sweeps['RMP'] = rmp
            
    df_grp = sweeps.groupby(['condition', 'QC']).mean().reset_index()

    
    for cond in sweeps.condition.unique():
        # Default to NaN
        results[cond] = np.nan
        
        # Filter the group for the current condition
        cond_group = df_grp[df_grp.condition == cond]
        if len(cond_group) > 1:
            # Try to get the 'included' RMP value
            included_value = cond_group[(cond_group.QC == 'included')]['RMP'].iloc[0]
            
            if pd.isna(included_value):
                # If 'included' RMP is NaN, then take the 'excluded' RMP
                excluded_value = cond_group[(cond_group.QC == 'excluded')]['RMP'].iloc[0]
                results[cond] = excluded_value
            else:
                # Use the 'included' RMP
                results[cond] = included_value
        else:
            # If there's only one QC value, just take it regardless of 'included' or 'excluded'
            results[cond] = cond_group['RMP'].iloc[0]
        
    return results, sweeps
    
def FI_and_APs(sweeps,dataset, rectype):
    results=pd.DataFrame()
    df2 = sweeps[(sweeps.stimulus_code.str.contains('teps')) & (sweeps.clamp_mode.values == 'CurrentClamp')].reset_index()
    if rectype == 'mono':
        if df2.shape[0] != 0:
            for index, row in df2.iterrows():
                sweep = dataset.sweep(sweep_number=row.sweep_number)
                if sweep.epochs['stim'] is not None:
                    #start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                    ext = SpikeFeatureExtractor()
                    res = ext.process(sweep.t, sweep.v, sweep.i)
                    curr = round(max(sweep.i,key=abs))
                    
                    if curr > 0:  # Ensure we proceed only if current is greater than zero
                        if len(res)==0:  # If res is empty, initialize it as an empty DataFrame
                            res = pd.DataFrame({'sweep_number': [row.sweep_number], 'cell': [cell], 'condition': [row.condition], 'curr_inj': [curr], 'clamp_mode': ['not_clamped' if pd.isnull(row.leak_pa) else 'clamped']})
                            sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr
                            sweeps.at[sweeps['sweep_number'] == row.sweep_number,'nAPs'] = 0
                        elif len(res)>0:
                            clamp_mode = 'not_clamped' if pd.isnull(row.leak_pa) else 'clamped'
                            res['sweep_number']=row.sweep_number
                            res['QC'] = row.QC
                            res['cell']=cell
                            res['condition']=row.condition
                            res['curr_inj']=curr
                            res['clamp_mode']=clamp_mode
                            res['ISI'] = res['threshold_t'].diff()
                            res['ISI'].fillna(0, inplace=True)
                            res['IF']=1/res['ISI']
                            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
                            labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']
                            # Use pd.cut() to categorize the 'IF' values into bins
                            res['f_bin'] = pd.cut(res['IF'], bins=bins, labels=labels, right=False, include_lowest=True)
                            sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr
                            sweeps.at[sweeps['sweep_number'] == row.sweep_number,'nAPs'] = res['threshold_t'].notnull().sum()
                        results = pd.concat([results, res], axis = 0)
        slopes={}
        for cond in sweeps.condition.unique():
            slopes[cond] = {'slope':np.nan,'rheobase': np.nan}
        for cond in sweeps.condition.unique():
            subdf = sweeps[(sweeps.condition == cond) & (sweeps.curr_inj < 500) & (sweeps.curr_inj > 0)]
            if subdf.shape[0] != 0:
                coeffs = np.polyfit(list(subdf['curr_inj']), list(subdf['nAPs']), 1)
                slope = coeffs[0]
                if len(subdf[subdf.nAPs !=0]) != 0:
                    slopes[cond] = {'slope':slope,'rheobase': subdf[subdf.nAPs >0].iloc[0]['curr_inj']}
                else:
                    slopes[cond] = {'slope':slope,'rheobase': sweeps[(sweeps.condition == cond)&(sweeps.stimulus_code.str.contains('teps'))&(sweeps.nAPs >0)].iloc[0]['curr_inj']}

    elif rectype == 'multi':
            print(rectype)
    return sweeps, results, slopes
       

def contains_substring(lst, sub):
    return any(sub in item for item in lst)

def check_conds(sweeps, substr):
    list1 = list(sweeps[sweeps.condition == 'baseline']['stimulus_code'].unique())
    list2= list(sweeps[sweeps.condition == 'naag']['stimulus_code'].unique())
    contains_list1 = contains_substring(list1, substr)
    contains_list2 = contains_substring(list2, substr)
    results = contains_list1 and contains_list2
    return results


def XY_LMM_correlation(df_file_metadata, Xcorrelate,Ycorrelate):
    df_file_metadata = df_file_metadata.dropna(subset=[Ycorrelate, Xcorrelate])
    df_file_metadata[Xcorrelate] = pd.to_numeric(df_file_metadata[Xcorrelate], errors='coerce')
    # LMM fitting
    md = smf.mixedlm(f"{Ycorrelate} ~ {Xcorrelate}", df_file_metadata, groups=df_file_metadata["NewLabel"])
    mdf = md.fit()
    print(mdf.summary())
    p_value = mdf.pvalues[Xcorrelate]
    # Annotate p-value
    print(p_value)
    return mdf.summary(), p_value


from scipy.stats import shapiro
def apply_normality_test(group, param):
    stat, p = shapiro(group[param])
    return pd.Series({'W': stat, 'p-value': p})
#%% proceed with analysis
logging.basicConfig(filename=f'error_log_{current_date}_v1.txt', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')


for cell in tqdm(to_analyse['cell'].unique(), desc='Processing cells', unit='cell'):    
    #load in the data
    try:
        # if overview[overview.cell == cell].status_code.iloc[0] == 'analysed':
        #     if overview[overview.cell == cell].experiment.iloc[0] == 'mEPSPs' or overview[overview.cell == cell].experiment.iloc[0] == 'sEPSPs':
        #         if overview[overview.cell == cell].status_igor.iloc[0] == 'analysed':
        #            overview.loc[overview['cell'] == cell, 'total_status'] = 'analysed'    
        #         elif pd.isna(overview[overview.cell == cell].status_igor.iloc[0]):
        #             overview.loc[overview['cell'] == cell, 'total_status'] = 'not yet'     
        #             overview.loc[overview['cell'] == cell, 'total_status'] = 'not_complete'     
                    
        #     print(f"Cell {cell} has already been analyzed. Skipping...")
        #     #continue
        # else:
            
            df=overview[overview.cell == cell]
            print(f' analysing cell {cell}')
            try:
                s=find_correct_file(cell, all_files)
            except Exception as e:
                print(f'file not existigng or wrong naming : {e}')
            # dataset, rectype, sweeps,hs=load_data_metadata(path, s, 0)
            # sweeps=assign_condition_based_on_index(df, sweeps)
            dataset, rectype = load_data(path, s)
            print ('data loaded')
    except Exception as e:
        # Log the error along with the cell identifier
        logging.error(f'Error processing cell {cell}: {str(e)}')

        # Optionally, you can mark these cells in your DataFrame to revisit later
        overview.loc[overview.cell == cell, 'status_code'] = 'error'
        overview.loc[overview.cell == cell, 'error'] = np.nan
        #continue

    sweeps=None
    #load in the metadata
    try:
        sweeps, hs = process_sweep_table(rectype, df, dataset, s, cell, 0)
        sweeps.to_csv((path+f'/all_data/sweeps/{cell}_sweeps.csv'))
        overview.at[overview.cell == cell, 'HS'] = hs
        print ('metadata loaded')
    except Exception as e:
        # Log the error along with the cell identifier
        logging.error(f'Error processing cell {cell}: {str(e)}')
        #continue
    
    #check which conditions you need to exclude and then assign a QC to the overview table
    if 'sweeps' != None:
        #assign QC code
        present_conditions = sweeps[sweeps.QC != 'excluded']['condition'].unique()
        if 'baseline' and 'naag' in present_conditions:
            qc = 'included'
            overview.at[overview.cell == cell, 'QC'] = qc
        else:
            qc = 'excluded'
            overview.at[overview.cell == cell, 'QC'] = qc
            overview.at[overview.cell == cell, 'status_code'] = 'analysed'
            overview.at[overview.cell == cell, 'error'] = np.nan
        print (f'file {cell} is {qc}')
        
    
        #QC probably not applicable to the beginning of the file. Check if useful for others (PS protocols)
        if len(sweeps[sweeps.Ra_norm == 1]) !=0 and sweeps[sweeps.Ra_norm == 1].iloc[0]['sweep_number'] !=0:
            protocol_keywords = '|'.join(['teps', 'hres', 'CHIRP', 'Search', 'Rheo', 'C2N', 'C2SS', 'TRIP', 'Ramp'])
            # Check if any protocol in the list is in the 'protocols' variable
            if sweeps['stimulus_code'].str.contains(protocol_keywords, regex=True, na=False).any():
                overview.loc[overview['cell'] == cell, 'sharable?'] = 'yes'
                print (f'data is sharable for cell {cell}')
            else:
                overview.loc[overview['cell'] == cell, 'sharable?'] = 'no'
                print (f'data is NOT sharable for cell {cell}')
            
            del protocol_keywords
    else:
        continue
        
    with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)        
            
    if qc == 'included':
        
        #get VC data irrespectful of protocol name
        try: 
            print (f'-------getting TP data for {cell}-----------')
            tp_results, sweeps = TP_VC_analysis(sweeps,dataset,rectype, hs)
            
            for cond, values in tp_results.items():
                    overview.loc[(overview['cell'] == cell) & (overview['condition'] == cond), 'Rin'] = values['rin']
                    overview.loc[(overview['cell'] == cell) & (overview['condition'] == cond), 'tau'] = values['tau']
                    overview.loc[(overview['cell'] == cell) & (overview['condition'] == cond), 'leak'] = values['leak']
                    print ('----------------------------------TP successful!!!!!!-----------')
            del values, cond
            with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                            overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)
        except Exception as e:
            # Log the error along with the cell identifier
            logging.error(f'Error processing cell {cell}: {str(e)}')
            print('no TP analysis was done')
            
            
            
        #get CC d unclamped data irrespectful of protocol name
        try:
            #get rmp from all non-clamped CC sweeps 
            print (f'-----------getting RMP data for {cell}-----------')
            rmp, sweeps = get_rmp(sweeps,dataset, rectype)
            for cond, rmp_value in rmp.items():
                # Use the condition to filter the DataFrame and assign the RMP values
                overview.loc[(overview['cell'] == cell) & (overview['condition'] == cond), 'RMP'] = rmp_value
                print ('----------------------------------RMP successful!!!!!-----------')
            del rmp_value, cond
            #update the overview with RMP, leak, tau and rin    
            with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)
        except Exception as e:
            # Log the error along with the cell identifier
            logging.error(f'Error processing cell {cell}: {str(e)}')
            print('no RMP analysis was done')
            
        protocols = sweeps.stimulus_code.unique()
        completed_protocols = set()    
        com = set()   
                
        for protocol in protocols:
            if 'Test' in protocol or 'pulse' in protocol:
                print(f'TP data already achieved : adding {protocol} to analysed')
                completed_protocols.add(protocol)
            if 'Stim' in protocol or 'espo' in protocol:
                print(f'stim_curve not needed : adding {protocol} to analysed')
                completed_protocols.add(protocol)
            if 'rmp' in protocol:
                print(f'rmp already achieved : adding {protocol} to analysed')
                completed_protocols.add(protocol)   
                
            #get spike data
            print (f'getting spike data for {cell}')    
            if 'steps' in protocol and 'TTX' not in str(df.recording_condition.iloc[0]):
                if 'spikes' not in globals():
                    cond_qc = check_conds(sweeps, 'teps')
                    if cond_qc is True:
                        print ('-------spike data present, analysing--------')
                        sweeps, spikes, slopes= FI_and_APs(sweeps,dataset,rectype)
                        sns.lineplot(data=sweeps, x='curr_inj', y='nAPs', hue='condition')
                        spikes.to_csv(path+f'/all_data/spikes/{cell}_spikes.csv')
                        for cond, values in slopes.items():
                            # Use the condition to filter the DataFrame and assign the RMP values
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == cond), 'FI_slope'] = values['slope']  
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == cond), 'Rheobase'] = values['rheobase']
                            print ('------------------------spikes successful!!!!!-----------')
                            
                #update the overview withspike data
                        with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                            overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)
                        print(f'steps analysed : adding {protocol} to analysed')
                        completed_protocols.add(protocol)
                else: 
                    completed_protocols.add(protocol)
                    print(f'steps already analysed : adding {protocol} to analysed')
            else: 
                completed_protocols.add(protocol)
                print(f'steps not used: adding {protocol} to analysed')
                    
                    
            print (f'getting stimulation data for {cell}')
            if 'stim_1x' in str(df.experiment.iloc[0]):
                stims_list = ['X3','X5','X06', 'X18', 'X7','20', '100', '40', '_2_5', '_5', '_10', '_20']
                if 'data_stim' not in globals():
                    print (f"protocol {protocol} has been analysed before in previous experiment; data taken from there")
                    stims=pd.read_csv('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment2_stimulation 5-20-100; excitability/results/20231101_lab_meeting/total_tables/lab2023-10-23newpeaks_total.csv')
                    istims=stims[stims.cell.str.contains(cell)]
                    if len(istims) == 0:
                        overview.loc[overview['cell'] == cell, 'status_stims'] = 'excluded'
                    else:
                        overview.loc[overview['cell'] == cell, 'status_stims'] = 'included'
                    data_stim={}
                    for cond in istims.condition.unique():
                        data_stim[cond] = {
                        'condition': cond.split('_')[0], 
                        'ppr': istims[(istims.condition == cond) & (istims.clamp_mode == 'CC')]['ppr'].iloc[0],
                        'rec': istims[(istims.condition == cond) & (istims.clamp_mode == 'CC')]['recovery'].iloc[0],
                        'stim_axis': istims[(istims.condition == cond) & (istims.clamp_mode == 'CC')]['axis_stim'].iloc[0],
                        'proximity_label': istims[(istims.condition == cond) & (istims.clamp_mode == 'CC')]['proximity_label'].iloc[0],
                        'stim_f': istims[(istims.condition == cond) & (istims.clamp_mode == 'CC')]['condition'].iloc[0].split('_')[1]}
                        
                    
                    for cond, values in data_stim.items():
                        if pd.notna(values['stim_f']):
                            stim_f = values['stim_f']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), f'ppr_{stim_f}'] = values['ppr']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), f'rec_{stim_f}'] = values['rec']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), 'stim_axis'] = values['stim_axis']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), 'stim_proximity'] = values['proximity_label']
                            print ('------------------------stims successful!!!!!-----------')
                        else:
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), f'ppr_{stim_f}'] = values['ppr']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), f'rec_{stim_f}'] = values['rec']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), 'stim_axis'] = values['stim_axis']
                            overview.loc[(overview['cell'] == cell) & (overview['condition'] == values['condition']), 'stim_proximity'] = values['proximity_label']
                            print ('------------------------stims successful!!!!!-----------')
                    
                    with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                            overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)
                            
                    print(f'-------stims found : adding {protocol} to analysed---------')
                    completed_protocols.add(protocol)        
                else:
                    completed_protocols.add(protocol)
                    print(f'stims already found : adding {protocol} to analysed')
             
            cond_qc = check_conds(sweeps, protocol[:5])
            if cond_qc is False:
                completed_protocols.add(protocol)
                print(f'{protocol} only in one condition; not analysed in this analysis')
            elif cond_qc is True:
                com.add(protocol)
            
             
        if completed_protocols == set(protocols) and str(df.experiment.iloc[0]) != 'mEPSPs' or str(df.experiment.iloc[0]) != 'sEPSPs' :
            overview.loc[overview['cell'] == cell, 'status_code'] =  'analysed'
            overview.loc[overview['cell'] == cell, 'status_igor'] =  'n/a'
            overview.loc[overview['cell'] == cell, 'total_status'] =  'complete'
            overview.loc[overview['cell'] == cell, 'error'] =  np.nan
            
            
            print(f"cell {cell} is fully analysed, and all the data is stored")
            
        elif completed_protocols != set(protocols) and  str(df.experiment.iloc[0]) == 'mEPSPs' or str(df.experiment.iloc[0]) == 'sEPSPs':
            overview.loc[overview['cell'] == cell, 'status_code'] =  'analysed'
            overview.loc[overview['cell'] == cell, 'status_igor'] =  'not yet'
            overview.loc[overview['cell'] == cell, 'total_status'] =  'not_complete'
            overview.loc[overview['cell'] == cell, 'error'] =  np.nan
                
            print(f"cell {cell} is code analysed, the data is stored, needs Igor analysis")    
        
        elif completed_protocols != set(protocols):
            overview.loc[overview['cell'] == cell, 'status_code'] =  'to be continued'
            overview.loc[overview['cell'] == cell, 'status_igor'] =  'not yet'
            overview.loc[overview['cell'] == cell, 'total_status'] =  'not_complete'
            overview.loc[overview['cell'] == cell, 'todo'] =  str(com)
       
        with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                            overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)
        sweeps.to_csv(path+f'/all_data/sweeps/{cell}_sweeps.csv')
        
        
        variables_to_delete = ['df', 's', 'dataset', 'rectype', 'sweeps', 'hs', 'tp_results', 'rmp', 'protocols', 'completed_protocols', 'spikes', 'slopes', 'stims_list', 'stims', 'istims', 'data_stim']
        for var in variables_to_delete:
            try:
                del globals()[var]
            except KeyError:
                print(f"{var} was not defined.")    
                

with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), 
                engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    overview.to_excel(writer, sheet_name= f'analysed_{current_date}', index=False)
 

#%%combine all exc data together
path_data = '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/clamped_excitability'
data_files= [f for f in os.listdir(path_data) if f.endswith('.csv')]

all_spikes = pd.DataFrame()
all_FI=pd.DataFrame()
all_slopes=pd.DataFrame()
for t in data_files:
    temp=pd.read_csv(f'{path_data}/{t}')
    if t.endswith('spikes.csv'):
        all_spikes = pd.concat([all_spikes,temp], axis=0, ignore_index = True)
    elif t.endswith('slopes.csv'):
        all_slopes = pd.concat([all_slopes,temp], axis=0, ignore_index = True)
    elif t.endswith('FI.csv'):
        all_FI= pd.concat([all_FI,temp], axis=0, ignore_index = True)

#get the 2xrheobase boxplot data

for cell in all_FI.cell.unique():
    temp=all_FI[all_FI.cell == cell]
    rheo = temp[temp.nAPs >0]['curr_inj'].iloc[0]
    rrheo = rheo*2
    #group = temp['group'].iloc[0]
    for cond in temp.condition.unique():
            n=temp[(temp.curr_inj == rrheo)&(temp.condition == cond)]['nAPs'].iloc[0]
            all_slopes.at[(all_slopes.cell == cell)&(all_slopes.condition == cond), '2xrheo'] = rrheo
            all_slopes.at[(all_slopes.cell == cell)&(all_slopes.condition == cond), '2xrheo_nAPs'] = n

all_spikes.to_csv(f'{path_data}/total_tables/all_spikes_{current_date}.csv')
all_FI.to_csv(f'{path_data}/total_tables/all_FI_{current_date}.csv')
all_slopes.to_csv(f'{path_data}/total_tables/all_slopes_{current_date}.csv')
        
#%%plot excitability

condition_palette = {'baseline': '#009FE3','naag': '#F3911A'}
#plot the data
groups=['Mouse','L2', 'L3']

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(30, 10))  # Adjust the size as needed
# Flatten the axes array for easier indexing
axes = axes.flatten()

for i, group in enumerate(groups):
    fi_data = all_FI[all_FI['group'] == group]
    spikes_data = all_spikes[all_spikes['group'] == group]
    slopes_data = all_slopes[all_slopes['group'] == group]

    # FI plot for the group
    sns.lineplot(data=fi_data, x='curr_inj', y='nAPs', ax=axes[i*2], hue='condition', palette=condition_palette)
    axes[i*2].set_title(f'{group} FI, N={len(fi_data.cell.unique())}')
    axes[i*2].set_xlabel('Current (pA)')
    axes[i*2].set_ylabel('# APs')
    axes[i*2].set_xlim([0, 400])
    
    print(f'{group}')
    print(AnovaRM(data=df, depvar='nAPs', subject='cell', within=['condition']).fit())

    
    # Threshold plot for the group
    order = ['First AP', '0-10', '10-20', '20-30', '30-40', '40-50']
    binned_spikes_data = spikes_data[spikes_data['f_bin'].isin(order)]
    sns.lineplot(data=binned_spikes_data, x='f_bin', y='threshold_v', ax=axes[i*2+1], hue='condition', palette=condition_palette)
    axes[i*2+1].set_title(f'{group} Binned V threshold, N={len(binned_spikes_data.cell.unique())}')
    axes[i*2+1].set_xlabel('IF bin')
    axes[i*2+1].set_ylabel('Threshold (mV)')
    
# Subplot 3: Boxplot Rheobase 
sns.boxplot(data=all_slopes,x='group', y='rheobase', ax=axes[7], hue='condition', order=groups,palette=condition_palette)  # Adjust variable names
axes[7].set_title('Rheobase')
axes[7].set_ylabel('Rheobase (pA)')

# Subplot 4: Boxplot Slope
sns.boxplot(data=all_slopes,x='group', y='slope', ax=axes[8], hue='condition', order=groups,palette=condition_palette)  # Adjust variable names
axes[8].set_title('FI slope')
axes[8].set_ylabel('FI slope')

# Subplot 5: Boxplot Rin
sns.boxplot(data=all_slopes,x='group', y='Rin', ax=axes[9], hue='condition', order=groups,palette=condition_palette)  # Adjust variable names
axes[9].set_title('Rin')
axes[9].set_ylabel('Input resistance (mOhm)')

# Subplot 6: Boxplot Rin
sns.boxplot(data=all_slopes,x='group', y='2xrheo_nAPs', ax=axes[10], hue='condition',order=groups, palette=condition_palette)  # Adjust variable names
axes[10].set_title(f'nAPs at 2x Rheobase')
axes[10].set_ylabel('# APs at 2x Rheobase')

axes[6].axis('off')
axes[11].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Effect of NAAG washin on excitability [clamped_data]')


#%%stats on excitability figure
def plot_XY_LMM_correlation(df, Xcorrelate, Xcorrelate_label, Ycorrelate, Ycorrelate_label):
    df = df.dropna(subset=[Ycorrelate, Xcorrelate])
    df[Xcorrelate] = pd.to_numeric(df[Xcorrelate], errors='coerce')



    # LMM fitting
    md = smf.mixedlm(f"{Ycorrelate} ~ {Xcorrelate}*condition", df, groups=df_file_metadata["group"])
    mdf = md.fit()
    print(mdf.summary())
    p_value = mdf.pvalues[Xcorrelate]
    predictions = mdf.predict(df_file_metadata.sort_values(Xcorrelate))
    

# Annotate p-value
if p_value < 0.00005:
    plt.annotate(f'p = {p_value:.2e}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
elif p_value < 0.0005:
    plt.annotate(f'p = {round(p_value, 5)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
elif p_value < 0.005:
    plt.annotate(f'p = {round(p_value, 4)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
elif p_value < 0.05:
    plt.annotate(f'p = {round(p_value, 3)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))
else:
    plt.annotate(f'p={round(p_value, 2)}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',bbox=dict(boxstyle="round", alpha=0.5, color='w'))



#%%check the PPR data
from PIL import Image

for cell in tqdm(overview.cell.unique(), desc='Processing cells', unit='cell'):
    exp=overview[overview.cell==cell]['experiment'].iloc[0]
    print(exp)
    if 'stim' in exp:
        peaks_cells.append(cell)
    else:
        no_peaks.append(cell)
        
peaks_path='/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment2_stimulation 5-20-100; excitability/results/20231101_lab_meeting/i_tables'


#%% peaks    
cell=peaks_cells[84]
print(cell)

cell_sweeps= pd.read_csv(os.path.join(path, 'all_data/sweeps', f'{cell}_sweeps.csv'))
df=cell_sweeps[((cell_sweeps.stimulus_code=='X06_baaseline')|(cell_sweeps.stimulus_code=='X3_baaseline')|(cell_sweeps.stimulus_code=='X5_naag')|(cell_sweeps.stimulus_code=='X18_naag'))&~(cell_sweeps.leak_pa.isna())&(cell_sweeps.QC=='included')]
try:
    cell_peaks=pd.read_csv(os.path.join(peaks_path, f'newpeaks{cell}.csv'))
except Exception as e:
    print(f'{e}')
    cell_peaks=pd.read_csv(os.path.join(peaks_path, f'allpeaks_{cell}.csv'))
print(len(df.condition.unique())>1)

    
dataset=create_ephys_data_set(nwb_file=f'{path}/all_data/{cell}.nwb')


#if there is no peaks
avg_baseline5=get_avg(dataset,cell_sweeps,'X06_baaseline','CurrentClamp', mode=0)
avg_naag5=get_avg(dataset,cell_sweeps,'X18_naag','CurrentClamp', mode=0)
peaks5=find_peaks(dataset,avg_baseline5,avg_naag5,5,f'{path}/all_data',file, path,cell_sweeps)

cell_peaks=peaks5
cell_peaks=cell_peaks.reset_index()


for cond in cell_peaks['condition'].unique():
    # Identify indices for the current condition
    indices = cell_peaks[cell_peaks['condition'] == cond].index
    
    # Compute base only once per condition
    base = cell_peaks.loc[indices, 'baseline_voltage'].iloc[0]
    
    # Calculate 'total_amplitude' directly in the original DataFrame
    cell_peaks.loc[indices, 'total_amplitude'] = cell_peaks.loc[indices, 'peak_voltage'] - base
    
    # Calculate 'synaptic_buildup' using the first 'total_amplitude' value of the current condition
    first_amplitude = cell_peaks.loc[indices, 'total_amplitude'].iloc[0]
    cell_peaks.loc[indices, 'synaptic_buildup'] = cell_peaks.loc[indices, 'total_amplitude'] / first_amplitude


plt.figure()
for protocol in df.stimulus_code.unique():
    if 'b' in protocol:
        cond='baseline_5'
    elif 'na' in protocol:
        cond='naag_5'
    avg=get_avg(dataset,cell_sweeps,protocol,'CurrentClamp', mode=0)
    peak1x,peak1y=cell_peaks[(cell_peaks.condition==cond)&(cell_peaks.peak_number==1)]['peak_index'].iloc[0],cell_peaks[(cell_peaks.condition==cond)&(cell_peaks.peak_number==1)]['peak_voltage'].iloc[0]
    peak2x,peak2y=cell_peaks[(cell_peaks.condition==cond)&(cell_peaks.peak_number==2)]['peak_index'].iloc[0],cell_peaks[(cell_peaks.condition==cond)&(cell_peaks.peak_number==2)]['peak_voltage'].iloc[0]
    plt.plot(avg)
    plt.plot([peak1x, peak2x], [peak1y, peak2y], marker='o', color='k')
                
cell_peaks.to_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/ppr/{cell}_updated_peaks.csv')
plt.savefig(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/ppr/{cell}_ppr_plot.eps', format='eps')
plt.savefig(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/ppr/{cell}_ppr_plot.png', format='png')
cell_sweeps.to_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/ppr/{cell}_sweeps.csv')   

plt.close('all')

#%% ppr data
total_path='/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/ppr'
ppr_files = [f for f in os.listdir(total_path) if f.endswith('updated_peaks.csv')]
all_ppr = pd.DataFrame()
for t in ppr_files:
    temp=pd.read_csv(f'{total_path}/{t}')
    cell=t[:-18]
    try:
        swps=pd.read_csv(f'{total_path}/{cell}_sweeps.csv')
    except Exception as e:
        try:
            swps=pd.read_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/all_data/sweeps/{cell}_sweeps.csv')
        except Exception as e:
            print(f'{e}')
            continue
            
    df=swps[((swps.stimulus_code=='X06_baaseline')|(swps.stimulus_code=='X3_baaseline')|(swps.stimulus_code=='X5_naag')|(swps.stimulus_code=='X18_naag'))&~(swps.leak_pa.isna())]
    if 'excluded' not in df.QC.unique(): 
        try:
            group = overview[overview.cell == cell]['group'].iloc[0]
        except Exception as e:
            print(f'{e}')
            group='tbd'
        temp['group']=group
        all_ppr = pd.concat([all_ppr,temp], axis=0, ignore_index = True)
    else:    
        continue
    
#%%
all_ppr5=all_ppr[(all_ppr.condition=='baseline_5')|(all_ppr.condition=='naag_5')]
#example cell data to plot explanation

example_cell='H23.29.245.11.64.01'
dataset=create_ephys_data_set(nwb_file=f'{path}/all_data/{cell}.nwb')
example_sweeps= pd.read_csv(os.path.join(path, 'all_data/sweeps', f'{cell}_sweeps.csv'))

df=example_sweeps[((example_sweeps.stimulus_code=='X06_baaseline')|(example_sweeps.stimulus_code=='X3_baaseline')|(example_sweeps.stimulus_code=='X5_naag')|(example_sweeps.stimulus_code=='X18_naag'))&~(example_sweeps.leak_pa.isna())&(example_sweeps.QC=='included')]
example_peaks=pd.read_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/ppr/{cell}_updated_peaks.csv')

avg_baseline=get_avg(dataset,example_sweeps,'X06_baaseline','CurrentClamp',down=1, mode=0)
avg_naag=get_avg(dataset,example_sweeps,'X18_naag','CurrentClamp', down=0,mode=0)


condition_palette = {'baseline_5': '#009FE3','naag_5': '#F3911A'}
#plot the data
groups=['Mouse','L2', 'L3']
n_mouse=len(all_ppr5[all_ppr5.group == 'Mouse']['cell'].unique())
n_l2=len(all_ppr5[all_ppr5.group == 'L2']['cell'].unique())
n_l3=len(all_ppr5[all_ppr5.group == 'L3']['cell'].unique())

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,15))  # Adjust the size as needed
# Flatten the axes array for easier indexing
axes = axes.flatten()

#in ax 0
plt.plot(avg_baseline, color=condition_palette['baseline_5'])
plt.plot(avg_naag, color=condition_palette['naag_5'])
axes[0].plot(x[peak1x, peak2x], [peak1y, peak2y], marker='o', color='k')
axes[0].plot([x1_ppr, x1_ppr], [y1_ppr, peak1y], marker='o', color='k')
   


sns.boxplot(data=all_ppr5, x='group', y='ppr', ax=axes[0], hue='condition',order=groups, palette=condition_palette)
axes[0].set_title('PPR')
axes[0].axhline(y=1, color='grey', linestyle='--')

sns.boxplot(data=all_ppr5, x='group', y='synaptic_buildup', ax=axes[1], hue='condition',order=groups, palette=condition_palette)
axes[1].set_title('Synaptic buildup')
axes[1].axhline(y=1, color='grey', linestyle='--')



plt.suptitle(f'Effect of NAAG washin on extracellular stimulation subthreshold response\n Mouse N={n_mouse}, L2 N={n_l2}, L3 N={n_l3}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


#%% get rmps
rmps = pd.DataFrame(columns = ['cell', 'QC', 'condition', 'new_rmp'])
for cell in all_sweeps.cell.unique():
    temp1 = all_sweeps.loc[all_sweeps.cell == cell]
    rmpi = temp1[(temp1.stimulus_code.str.contains('rmp')) |(temp1.stimulus_code.str.contains('RMP'))]
    if len(rmpi) > 0:
        print(cell)
        print(rmpi)            



#%% stim results
data=df[(df.QC == 'included')&((df.condition == 'baseline') | (df.condition == 'naag'))]
params=['ppr_5', 'ppr_2', 'ppr_10', 'ppr_20', 'rec_5', 'rec_10', 'rec_2', 'rec_20']
groups = ['Mouse', 'L2', 'L3']
xes=['stim_axis', 'stim_proximity']

for param in params: 
    for group in groups:
        data_plot = data[data.group == group]
        plt.figure()
        for x in xes:
            if x == 'stim_proximity':
                order =  ['Proximal', 'Distal']
            elif x == 'stim_axis':
                order = ['basal', 'apical']
            plt.figure()
            sns.boxplot(data=data_plot, x=x, y=param, hue = 'condition',palette = condition_palette, order = order )
            plt.title(group)
            #plt.savefig(path+f'/all_data/eps/{param}_{x}_{group}.eps')




#%% get the Ns and stats for the plots
def determine_species(group):
    if group == 'Mouse':
        return 'Mouse'
    elif group in ['L2', 'L3']:
        return 'Human'
    else:
        return 'Unknown'


import scipy.stats as stats

def box_stats(data,param, group):
    info=data.loc[(data.group==group) & (~data[param].isna())]
    bsl = info[info.condition == 'baseline'][param]
    naag = info[info.condition == 'naag'][param]
    n=len(bsl)
    
    if (len(bsl)>7) & (len(naag)>7):
        normal = (stats.normaltest(bsl)[1]>0.05) & (stats.normaltest(naag)[1]>0.05)
    else:
        normal = False
    if normal:
            test='paired t_test'
            equal_var = stats.bartlett(bsl, naag)[1]>0.05
            res=stats.ttest_ind(bsl, naag, equal_var=equal_var)
            pval  = res[1]
            test_info='t-test, p=' + np.format_float_scientific(pval, precision=1) + ', stat = ' + np.format_float_scientific(res[0], precision=3)
            metric_info = 'mean+-SD: baseline ' + np.format_float_positional(bsl.mean(), precision=4) + '+-' + np.format_float_positional(bsl.std(), precision=4) + ' naag ' + np.format_float_positional(naag.mean(), precision=4) + '+-' + np.format_float_positional(naag.std(), precision=4)
    else:
        test='Wilcoxon rank_test'
        res  = stats.ranksums(bsl, naag)
        pval  = res[1]
        test_info='ranksums, p=' + np.format_float_scientific(pval, precision=1) + ', stat = ' + np.format_float_scientific(res[0], precision=3)
        metric_info = 'median(Q1-Q3): baseline ' + np.format_float_positional(bsl.median(), precision=4) + '(' + np.format_float_positional(np.quantile(bsl, 0.25), precision=4) + '-' + np.format_float_positional(np.quantile(bsl, 0.75), precision=4) + ') naag ' + np.format_float_positional(naag.median(), precision=4) + '(' + np.format_float_positional(np.quantile(naag, 0.25), precision=4) + '-' + np.format_float_positional(np.quantile(naag, 0.75), precision=4) + ')'
    if pval < 0.001:
        stars='***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = 'N.S.'
        
    results=pd.DataFrame()
    results=results.append({'group':group,'param':param,'n' : n,'test':test,'test_info':test_info,'pval':pval, 'stars':stars}, ignore_index=True)    
    return results

all_stats=pd.DataFrame()
for group in all_slopes.group.unique():
    resi=box_stats(all_slopes,'slope', group)
    all_stats=pd.concat([all_stats,resi], axis=0, ignore_index = True)


sns.boxplot(data=sags,x='Group', y='sag_ratio', hue='condition', order=groups,palette=condition_palette)  # Adjust variable names

sags=pd.read_csv('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment2_stimulation 5-20-100; excitability/results/20231101_lab_meeting/total_tables/lab2023-10-11_allsags_total.csv')
#%% process the continuous RMP data
palette = {'baseline': '#009FE3','mini_baseline': '#009FE3','naag': '#F3911A', 'vehicle': '#5C120F', 'naag (50uM)':'#F3911A',}

rmp_sweeps=sweeps[(sweeps.stimulus_code.str.contains('ttx')) | (sweeps.stimulus_code.str.contains('rmp'))]
current_x = 0
for index, row in rmp_sweeps.iterrows(): 
    if rectype == 'mono':
        swpnr=row['sweep_number']
        sweep=dataset.sweep(sweep_number=swpnr)
        swp=sweep.v
        inp=sweep.i
        ext = SpikeFeatureExtractor()
        results = ext.process(sweep.t, swp, inp)
        indexes= np.where(inp !=0)[0]
                
        swp[indexes] = np.nan
        cond=row['condition']
        
        sweep_length = len(swp)
        x_values = range(current_x, current_x + sweep_length)
    elif rectype == 'multi':
        swpnr=row['sweep_number']
        cond=row['condition']
        f=h5py.File(f'{path}/all_data/{s}', 'r')
        swp=f['acquisition'][row.key_name]['data']
        sweep_length = len(swp)
        x_values = range(current_x, current_x + sweep_length)
        
    color=palette.get(cond, 'k')     
    plt.plot(x_values, swp, color=color)#, #label=cond if i == 0 else "")
    # Update the current x-axis position for the next trace
    current_x += sweep_length
plt.title(f'{cell}')

#%% get rmp vehicles

import re
def find_filename(cell, all_files):
    parts = cell.split('.')
    main_part = '.'.join(parts[:-1])  # Take all except the last part
    last_two_digits = parts[-1]
    
    # Format last_two_digits to ensure two digits
    last_two_digits = f"{int(last_two_digits):02d}"
    
    # Prepare a pattern that checks both possibilities for the last two digits being at start or end of the numeric sequence
    pattern = re.compile(rf'{main_part}\.({last_two_digits})(\d{{2}})?\.nwb$|{main_part}\.(\d{{2}})({last_two_digits})\.nwb$')
    
    for filename in all_files:
        if pattern.search(filename):
            return filename



#all_cells = overview[overview.experiment.str.contains('rmp')].cell.unique()
palette = {'baseline': '#009FE3','mini_baseline': '#009FE3','naag': '#F3911A', 'vehicle': '#5C120F', 'naag (50uM)':'#F3911A',}

data=pd.DataFrame()
for cell in all_cells:
    sweeps=pd.read_csv(path+f'/all_data/sweeps/{cell}_sweeps.csv')
    df=overview[overview.cell == cell]
    rmp_sweeps=sweeps[(sweeps.stimulus_code.str.contains('spont')) | (sweeps.stimulus_code.str.contains('rmp'))]
    if 'key_name' not in rmp_sweeps.columns:
        dataset=create_ephys_data_set(nwb_file=path+f'/all_data/{cell}.nwb')
        
    for index, row in rmp_sweeps.iterrows(): 
        if 'key_name' not in rmp_sweeps.columns:
            swpnr=row['sweep_number']
            sweep=dataset.sweep(sweep_number=swpnr)
            swp=sweep.v
            cond=row['condition']
            data= data.append({'cell':cell,'sweep_number': swpnr,'condition': cond, 'median_rmp': np.mean(swp)}, ignore_index=True)
        else:
            s = find_filename(cell, all_files)
            swpnr=row['sweep_number']
            cond=row['condition']
            f=h5py.File(f'{path}/all_data/{s}', 'r')
            swp=f['acquisition'][row.key_name]['data']
            data= data.append({'cell':cell,'sweep_number': swpnr,'condition': cond, 'median_rmp': np.mean(swp)}, ignore_index=True)
            
data['sweep_order'] = data.groupby('cell')['sweep_number'].transform(lambda x: x - x.min() + 1)
data['rmp_norm'] = data.groupby('cell')['median_rmp'].transform(lambda x: -x/ x.iloc[0])
data.to_csv(f'{path}/all_data/rmp/{cell}_rmp.csv')
sns.boxplot(data=data, x='sweep_order', y='rmp_norm', hue='condition')
#%% get ssteps vehicles
spks = [f for f in os.listdir(os.path.join(path, 'all_data/spikes')) if '268' in f]

all_spikes=pd.DataFrame()
for f in spks:
    temp=pd.read_csv(path+f'/all_data/spikes/{f}')
    all_spikes=pd.concat([all_spikes, temp], axis=0)


order=['First AP', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']
plt.figure()
sns.lineplot(data=all_spikes, x='curr_inj', y='nAPs', hue='condition')
plt.figure()
sns.boxplot(data=all_spikes, x='f_bin', y='threshold_v', hue='condition',order = order)

plt.figure()
sns.lineplot(data=all_spikes, x='f_bin_numeric', y='threshold_v', hue='condition')


bins=['First AP', '10-20', '20-30', '30-40', '40-50']
category_to_integer = {
    '': 0, 'First AP': 1, '0-10': 2, '10-20': 3, '20-30': 4, '30-40': 5,
    '40-50': 6, '50-60': 7, '60-70': 8, '70-80': 9, '80-90': 10,
    '90-100': 11, '100+': 12}
plot=all_spikes[all_spikes.f_bin.isin(bins)]
plot['f_bin_numeric'] = plot['f_bin'].map(category_to_integer)

plt.figure()
sns.boxplot(data=plot, x='f_bin', y='threshold_v', hue='condition',order = order)

plt.figure()
sns.lineplot(data=plot, x='f_bin_numeric', y='threshold_v', hue='condition')


#%%

cell='H24.29.274.11.63.01'
df=overview[overview.cell == cell]
dataset, rectype = load_data(path, s)
s=[f for f in all_files if cell[:-3] in f][0]
s
dataset, rectype = load_data(path, s)
cell_sweeps = pd.read_csv(os.path.join(path, 'all_data/sweeps', f'{cell}_sweeps.csv'))
df2 = cell_sweeps[(cell_sweeps.stimulus_code.str.contains('teps')) & 
         ~(cell_sweeps.leak_pa.isna()) & 
         (cell_sweeps.QC == 'included') & 
         (cell_sweeps['condition'].isin(['baseline', 'naag']))]

df2['group']=overview[overview.cell == cell]['group'].iloc[0]

df2 = cell_sweeps[(cell_sweeps.stimulus_code.str.contains('teps')) & 
         ~(cell_sweeps.leak_pa.isna()) & 
         (cell_sweeps.QC == 'included') & 
         (cell_sweeps['condition'].isin(['baseline', 'naag']))]

df2['group']=overview[overview.cell == cell]['group'].iloc[0]
cell_spikes=pd.DataFrame()
if len(df.condition.unique())>1:
    if df2.shape[0] != 0:
        for index, row in df2.iterrows():
            try:
                sweep = dataset.sweep(sweep_number=row.sweep_number)
                if sweep.epochs['stim'] is not None:
                    #start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                    ext = SpikeFeatureExtractor()
                    res = ext.process(sweep.t, sweep.v, sweep.i)
                    curr = round(max(sweep.i,key=abs))
                    if curr < 0:  # Ensure we proceed only if current is greater than zero
                        try:
                            start,end=sweep.epochs['stim'][0],sweep.epochs['stim'][1]
                            fs=sweep.sampling_rate
                            v=sweep.v
                            vss=v[int(end - fs/10):int(end-1)].mean()
                            vbase=v[int(start - fs/10):int(start-1)].mean()
                            rin=(vss-vbase)/curr*1000
                            df2.at[df2['sweep_number'] == row.sweep_number,'Rin'] = rin
                            df2.at[df2['sweep_number'] == row.sweep_number, 'curr_inj'] = np.nan
                            df2.at[df2['sweep_number'] == row.sweep_number,'nAPs'] = np.nan
                        except:
                            df2.at[df2['sweep_number'] == row.sweep_number,'Rin'] = np.nan
                            df2.at[df2['sweep_number'] == row.sweep_number, 'curr_inj'] = np.nan
                            df2.at[df2['sweep_number'] == row.sweep_number,'nAPs'] = np.nan 
                    elif curr > 0:
                        if len(res)==0:  # If res is empty, initialize it as an empty DataFrame
                                res = pd.DataFrame({'sweep_number': [row.sweep_number], 'cell': [cell],'group':[row.group], 'condition': [row.condition], 'curr_inj': [curr], 'clamp_mode': ['not_clamped' if pd.isnull(row.leak_pa) else 'clamped']})
                                df2.at[df2['sweep_number'] == row.sweep_number, 'curr_inj'] = curr
                                df2.at[df2['sweep_number'] == row.sweep_number,'nAPs'] = 0
                        elif len(res)>0:
                                clamp_mode = 'not_clamped' if pd.isnull(row.leak_pa) else 'clamped'
                                res['sweep_number']=row.sweep_number
                                res['QC'] = row.QC
                                res['cell']=cell
                                res['group']=row.group
                                res['condition']=row.condition
                                res['curr_inj']=curr
                                res['clamp_mode']=clamp_mode
                                res['ISI'] = res['threshold_t'].diff()
                                res['ISI'].fillna(0, inplace=True)
                                res['IF']=1/res['ISI']
                                bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
                                labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']
                                # Use pd.cut() to categorize the 'IF' values into bins
                                res['f_bin'] = pd.cut(res['IF'], bins=bins, labels=labels, right=False, include_lowest=True)
                                res['f_bin'] = res['f_bin'].cat.add_categories(['First AP'])
                                res.at[0, 'f_bin'] = 'First AP'    
                                df2.at[df2['sweep_number'] == row.sweep_number, 'curr_inj'] = curr
                                df2.at[df2['sweep_number'] == row.sweep_number,'nAPs'] = res['threshold_t'].notnull().sum()
                        
                        cell_spikes = pd.concat([cell_spikes, res], axis = 0)               
            except Exception as e:
                print(f"Error processing sweep {row.sweep_number} for {cell}: {e}")
data = []

# Loop through each unique condition
for cond in df['condition'].unique():
    try:
        # Filter the DataFrame for the current condition and valid current injections
        subdf = df2[(df2['condition'] == cond) & (df2['curr_inj'] < 500) & (df2['curr_inj'] > 0)]
        group = df2.iloc[0]['group'] 
        if not subdf.empty:
            # Perform linear regression to get the slope
            coeffs_slope = np.polyfit(subdf['curr_inj'], subdf['nAPs'], 1)
            slope = coeffs_slope[0]
            coeffs_rin = np.polyfit(subdf['curr_inj'], subdf['nAPs'], 1)
            rin = coeffs_rin[1]
            
            # Find the first sweep where an AP occurs
            first_ap_df = subdf[subdf['nAPs'] > 0]
            rheobase = first_ap_df.iloc[0]['curr_inj'] if not first_ap_df.empty else np.nan
            
            # Append the result to the list as a dictionary
            data.append({'cell': cell, 'group': group, 'condition': cond, 'rheobase': rheobase, 'slope': slope, 'Rin': rin})
        else:
            # Append a row with NaNs if no data is found
            data.append({'cell': np.nan, 'group': np.nan, 'condition': cond, 'rheobase': np.nan, 'slope': np.nan, 'Rin': np.nan})
    except Exception as e:
        print(f"Error processing condition {cond} for cell {cell}: {e}")

# Convert the list of dictionaries to a DataFrame
slopes = pd.DataFrame(data)


#save slopes, spikes and FI; write into overview that it is done, savce the overview                
overview.at[overview.cell == cell, 'clamped_FI'] = 'done'    
overview.at[overview.cell == cell, 'status_code'] = 'analysed'  
overview.at[overview.cell == cell, 'status_igor'] = 'n/a'  
overview.at[overview.cell == cell, 'total_status'] = 'complete'  
with pd.ExcelWriter(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    overview.to_excel(writer, sheet_name=f'analysed_{current_date}', index=False)        

slopes.to_csv(os.path.join(path, 'clamped_excitability', f'{cell}_slopes.csv'))
df2.to_csv(os.path.join(path, 'clamped_excitability', f'{cell}_FI.csv'))  
cell_spikes.to_csv(os.path.join(path, 'clamped_excitability', f'{cell}_spikes.csv'))



# sweeps = pd.read_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/analysis/results/QC_files/sweeps_cleared_{cell}.csv')
# #getting the set numbe into the spikes table for stim_xxx
# mrg=pd.read_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/analysis/results/i_data/{cell}/total_spikes_{cell}.csv')
# mrg_unique = mrg.drop_duplicates(subset=['sweep_number'], keep='first')
# # Create the dictionary
# sweep_to_set_dict = pd.Series(mrg_unique['set_number'].values, index=mrg_unique['sweep_number']).to_dict()

# spikes['set_number'] = spikes['sweep_number'].map(sweep_to_set_dict)
# spikes['set_number'] = spikes['set_number'].fillna(method='bfill')

# #getting the set numbe into the srims table for stim_xxx
# istims = pd.read_csv(f'/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/analysis/results/i_data/{cell}/stim_{cell}.csv')   


# gr_istims  = gr_istims = istims.groupby(['condition', 'set_number', 'frequency']).first().reset_index()
# for index, row in gr_istims.iterrows():
#     gr_istims.loc[index, 'conditionx'] = row.condition + "_"+str(row.set_number) + '_'+ str(row.frequency)
    
# for index, row in istims.iterrows():
#     istims.loc[index, 'conditionx'] = row.condition + "_"+str(row.set_number) + '_'+ str(row.frequency)    

# data_stim={}
# for cond in gr_istims.conditionx.unique():
    
#     data_stim[cond] = {
#     'condition': cond.split('_')[0], 
#     'set_number' : cond.split('_')[1], 'stim_f': cond.split('_')[2],
#     'ppr': istims[(istims.conditionx == cond) & (istims.peak_number == 2)]['total_amplitude_normto1'].iloc[0],
#     'rec': istims[(istims.conditionx == cond) & (istims.peak_number == 6)]['total_amplitude_normto1'].iloc[0]}
    
# #'peak_ratios'
    

# ppr_data = istims[(istims.peak_number == 2)].copy()
# ppr_data['ppr'] = ppr_data['total_amplitude_normto1']
# ppr_data = ppr_data.groupby(['condition', 'frequency'])['ppr'].mean().reset_index()

# rec_data = istims[(istims.peak_number == 6)].copy()
# rec_data['rec'] = rec_data['total_amplitude_normto1']
# rec_data = rec_data.groupby(['condition', 'frequency'])['rec'].mean().reset_index()

# for index, row in ppr_data.iterrows():
#     overview.loc[(overview['cell'] == cell) & (overview['condition'] == row.condition), f'ppr_{str(int(row.frequency))}'] = row.ppr
# for index, row in rec_data.iterrows():
#     overview.loc[(overview['cell'] == cell) & (overview['condition'] == row.condition), f'rec_{str(int(row.frequency))}'] = row.rec

# #mrg_unique = mrg.drop_duplicates(subset=['sweep_number'], keep='first')     - useful for first AP   
            