#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:17:00 2025

@author: annagalakhova
"""
import toolkit
from toolkit import *
from datetime import datetime, timedelta
from ipfx.dataset.create import create_ephys_data_set
from ipfx.feature_extractor import SpikeFeatureExtractor
import pandas as pd
import re
import numpy as np
import scipy as sp
from scipy.signal import butter, filtfilt
import os
import matplotlib.pyplot as plt
import h5py
# import toolkit
#from toolkit import *
import seaborn as sns
from datetime import datetime
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
current_date=datetime.now().date()
%matplotlib qt

#%% general directories
path= '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project'
try:
    overview = pd.read_excel(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), sheet_name=f'analysed_{current_date}')
except:
    overview = pd.read_excel(os.path.join(path, 'all_data', 'all_data_overview.xlsx'), sheet_name=-1)
    
all_cells = overview.cell.unique()
all_files= [f for f in os.listdir(os.path.join(path, 'all_data')) if f.endswith('.nwb')]
savepath='/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/final_data_assesment'
condition_palette={'baseline': '#009FE3','naag': '#F3911A', 'vehicle': '#706F6F', 'washout':'#6CB52D', 'wash':'#B47C46'}
#%%functions 
def find_correct_file(cell, all_files):
    """
    Finds the correct file from a list of files based on a given 'cell' name, in case cell is in a multichannel file
    """
    # First, look for an exact match (including .nwb)
    exact_matches = [f for f in all_files if f.startswith(cell)]
    if exact_matches:
        return exact_matches[0]  # Return the first exact match
    # If no exact match, use partial matching as a fallback
    partial_matches = [f for f in all_files if cell[:17] in f and cell[17:] in f[17:-4]]  # Match without .nwb
    if partial_matches:
        return partial_matches[0]  # Return the first partial match
    
#support functions for load data  
def n_cell_in_file(s):
    cells = re.findall(r'0\d', s.split('.')[-2]) #detect the part in naming of file which represents cells
    slice_name= s.split('.')[:-2]
    full_names=['.'.join(slice_name+[i]) for i in cells]
    return len(cells), full_names
  
#load in the data - either an ipfx dataset or an h5py in case of multichannel data or an error
def load_data(path, cell, all_files):
    s=find_correct_file(cell, all_files)
    n_cells=n_cell_in_file(s)[0]
    #load in the data
    if n_cells == 1:
        try:
            dataset = create_ephys_data_set(nwb_file=f'{path}/all_data/{s}')
            rectype = 'mono'
        except Exception as e:
            print(f'Potentially recording has multiple channels : {e}')
            try:
                rectype = 'damaged'  
                dataset= h5py.File(f'{path}/all_data/{s}', 'r')
            except Exception as e:
                print(f'cannot load the data : {e}')    
    else:# If the error is related to "tuple", switch to using h5py
        rectype = 'multi'
        dataset= h5py.File(f'{path}/all_data/{s}', 'r')
    return dataset, rectype

#support function to extend the QC to other types of parameters
def extend_qc_check(df):
    """
    Extends QC checks by adding QC_extended column
   
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    # Initialize QC_extended column
    df['QC_extended'] = df['QC']
    # Process row by row to check conditions
    for idx, row in df.iterrows():
        if pd.notna(row['Ra']):  # Only process rows where Ra exists
            if row['QC'] == 'included':
                df.at[idx, 'QC_extended'] = 'included'
            else:
                # Check Ra conditions
                if pd.notna(row['Ra']) and row['Ra'] < 20:
                    df.at[idx, 'QC_extended'] = 'included'
                else:
                    df.at[idx, 'QC_extended'] = 'excluded'
                
                # Check leak current if it exists
                if pd.notna(row['leak_pa']) and row['leak_pa'] < -300:
                    df.at[idx, 'QC_extended'] = 'excluded'
    # Forward fill QC_extended until next valid Ra measurement
    last_valid_qc = None
    for idx, row in df.iterrows():
        if pd.notna(row['Ra']):
            last_valid_qc = df.at[idx, 'QC_extended']
        elif last_valid_qc is not None:
            df.at[idx, 'QC_extended'] = last_valid_qc
    return df

#convert unix time to readable time
def convert_timestamp_to_human(timestamp):
    igor_to_unix_offset = 2082844800
    unix_timestamp=timestamp-igor_to_unix_offset
    output=datetime.fromtimestamp(unix_timestamp)
    return output

#split the condition to smaller ones if needed
def split_single_condition(condition):
       if pd.isna(condition):
           return '', ''
       if '_' in condition:
           main, state = condition.split('_', 1)  # split on first underscore
           return main, state
       return condition, ''

#load in the sweep metadata 
def process_mono_sweep_table(dataset,s,cell,overview,ref=0):
    df=overview[overview.cell == cell]
    beg_swp=df.fswp_condition.iloc[0]
    #get the sweep table
    sweeps=dataset.filtered_sweep_table() 
    sweeps['cell']= cell
    sweeps['group'] = df.group.iloc[0]
    sweeps=sweeps[beg_swp:]
    f=h5py.File(f'{path}/all_data/{s}', 'r')
    #check which HS to use
    unique_channels = set(name.split('_')[-1] for name in f['acquisition'].keys())
    if len(unique_channels) == 1:
        hs_nr_from_f = int(next(iter(unique_channels))[-1])  # Extract from f['acquisition']
        print(f'Identified channel from acquisition: {hs_nr_from_f}')
    else:
        print(f'Cannot find a unique channel in acquisition for cell {cell}')
        hs_nr_from_f = None
    if not pd.isna(df.HS.iloc[0]):
        hs_nr_from_df=int(df.HS.iloc[0])
        print(f'HS from DataFrame: {hs_nr_from_df}')
        if hs_nr_from_f is not None:
            # Compare the two HS values
            if hs_nr_from_df == hs_nr_from_f:
                hs_nr = hs_nr_from_df  # Use the value from DataFrame if they match
                print(f'HS values match: using {hs_nr}')
            else:
                hs_nr = hs_nr_from_f  # Use the value from acquisition if they don't match
                print(f'HS mismatch: using channel from acquisition ({hs_nr}) and updating overview')
                overview.at[overview.cell == cell, 'HS'] = hs_nr
        else:
            hs_nr = hs_nr_from_df  # If no unique channel from acquisition, default to DataFrame
            print(f'Using HS from DataFrame: {hs_nr}')
    else:
    # If DataFrame value is missing, rely on acquisition channel if available
        if hs_nr_from_f is not None:
            hs_nr = hs_nr_from_f
            print(f'HS from DataFrame is missing: using channel from acquisition ({hs_nr}) and updating overview')
            overview.at[overview.cell == cell, 'HS'] = hs_nr
        else:
            print(f'Cannot determine HS for cell {cell}: both sources are ambiguous or missing')
            hs_nr = None  # Mark as missing
    #locate the Rm and Ra
    for index,row in sweeps.iterrows():
        print(f'Processing sweep {row.sweep_number} - adding Ra and Rm')
        sweeps.at[index, 'Rm'] = np.nan
        indexx = f'{row.sweep_number:05d}'
        #locate capacitance compensation 
        if 'whole_cell_capacitance_comp' in f['acquisition'][f'data_{indexx}_AD{hs_nr}'].keys(): 
            sweeps.at[index, 'compensation'] = 'yes'
        else:
            sweeps.at[index, 'compensation'] = 'no'
        #calculate for the ones you couldn't locate if from compensating
        try:
            sweep = dataset.sweep(row.sweep_number)
            fs=sweep.sampling_rate
            #detect the testpulse
            try:
                stimulus_changes = np.where(sweep._stimulus[:-1] != sweep._stimulus[1:])[0]
            except Exception as e:
                print(f'Could not find the TestPulse : {e}')
                tp_start, tp_end = None
                ra = np.nan
                sweeps.at[index, 'Ra'] = ra
                use_h5py = False
            if len(stimulus_changes) < 2:
                sweeps.at[index, 'Ra'] = np.nan
                sweeps.at[index, 'compensation'] = 'no'
                print(f"No TP epoch found for sweep {row.sweep_number}")
                pass
            else:
                tp_start =stimulus_changes[0] + 1
                tp_end = stimulus_changes[1] + 1
                use_h5py = False
            if len(sweep._stimulus) > tp_end:
                v=sweep._stimulus[: tp_end]
                i=sweep._response[: tp_end]
            else: 
                print(f'incomplete sweep {row.sweep_number}')
                ra = np.nan
                sweeps.at[index, 'Ra'] = ra
                use_h5py = False
        except Exception as e:
             print(f'Could not open the data with h5py for TP correction : {e}')
             use_h5py = True
        if use_h5py:     
            sweepi = f['acquisition'][f'data_{indexx}_AD{hs_nr}']['data']
            sweepv=f['stimulus']['presentation'][f'data_{indexx}_DA{hs_nr}']['data']
            fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
            tp_end = ((np.where(sweepv[:-1] != sweepv[1:])[0]) + 1)[1]
            v=sweepv[: tp_end]
            i=sweepi[: tp_end]
        #get the values to get Ra and tau
        try:
            bsl = np.mean(i[tp_start-int(fs*0.002):tp_start-int(fs*0.001)])
            peak, peak_id = max(i, key=abs), np.argmax(np.abs(i))
            tp_inp= round(max(v, key=abs))
            if row.clamp_mode == "VoltageClamp":
                ra=tp_inp/(peak-bsl)*1000
                if ra > 0:
                    sweeps.at[index, 'Ra'] = ra
                else: 
                    sweeps.at[index, 'Ra'] = np.nan
                #calculate tau
                y=i[peak_id:]
                x = np.arange(0, len(y))
                m_guess = 500
                b_guess = -100
                t_guess = 0.03
                if len(y) !=0:
                    try:
                        (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess],maxfev=100000)
                        tau=-1/(-t_true)/fs*1000
                        sweeps.at[index, 'tau'] = tau
                    except Exception as e:
                         print(f'could not fit the tau for sweep {row.sweep_number} : {e}')
                         tau=np.nan
                         sweeps.at[index, 'tau'] = tau
            elif row.clamp_mode == "CurrentClamp":
                rm=(tp_inp/(peak-bsl))
                sweeps.at[index, 'Rm'] = rm
        except Exception as e:
            print(f'Could not calculate Rm and Ra for sweep {sweep} : {e}')
            sweeps.at[index, 'compensation'] = 'no'
            sweeps.at[index, 'Ra'] = np.nan
            sweeps.at[index, 'tau'] = np.nan
            sweeps.at[index, 'Rm'] = np.nan
    #normalise Ra onto the first value unless otherwise specified
    #Assign Ra_norm
    if 'Ra' in sweeps.columns and not sweeps['Ra'].dropna().empty:
        if len(sweeps[sweeps.clamp_mode == 'VoltageClamp']) != 0:
            ra_norm_factor = sweeps['Ra'].dropna().iloc[ref] #choose here you first reference Ra
            sweeps['Ra_norm'] = sweeps['Ra'] / ra_norm_factor
        else:
            print("No valid 'Ra' values found. Defaulting ra_norm_factor to 1.")
    else:
        ra = np.nan
        sweeps.at[index, 'Ra'] = ra  
        sweeps.at[index, 'Ra_norm'] = np.nan
    #add timestamp for further check offline
    digitiser = list(f['general']['labnotebook'].keys())[0]
    for i in range(0,len(f['general']['labnotebook'][digitiser]['textualValues'])):
        temp=f['general']['labnotebook'][digitiser]['textualValues'][i]
        sweep_number=int(temp[0][0])
        timestamp=float(temp[2][0])
        readable_timestamp=convert_timestamp_to_human(timestamp)
        sweeps.at[sweeps['sweep_number'] == sweep_number, 'date'] = readable_timestamp.date()
        sweeps.at[sweeps['sweep_number'] == sweep_number, 'time'] = readable_timestamp.time().strftime('%H:%M')
        sweeps.at[sweeps['sweep_number'] == sweep_number, 'time_unix'] = timestamp
    #determine the experiment type
    unique_conditions= df['condition'].unique()
    if 'vehicle' in unique_conditions:
        sweeps['exp_type'] = 'control'
    elif 'naag' in unique_conditions:
        sweeps['exp_type'] = 'experiment'
    #add conditions from the overview reference table        
    for condition in unique_conditions: 
        start_index = df[df['condition'] == condition]['fswp_condition'].iloc[0]
        sweeps.at[start_index:,'condition'] = condition
    #separate the washin_states for cells where it is applicable
    for idx, row in sweeps.iterrows():
        main, state = split_single_condition(row['condition'])
        sweeps.at[idx, 'condition'] = main
        sweeps.at[idx, 'wash_in_state'] = state
    #add QC per sweep (based on Ra)
    qckech = np.nan
    sweeps['QC'] = qckech
    sweeps.loc[(sweeps['Ra_norm'] >= 0.7) & (sweeps['Ra_norm'] <= 1.3), 'QC'] = 'included'
    sweeps.loc[(sweeps['Ra_norm'] < 0.7) | (sweeps['Ra_norm'] > 1.3) | (sweeps['Ra'] > 20), 'QC'] = 'excluded'
    sweeps['QC'].fillna(method='ffill', inplace=True)
    sweeps['QC'].fillna(method='bfill', inplace=True)
    #apply extended QC (QC change is too high, but Ra absolute is below 20)
    sweeps=extend_qc_check(sweeps)
    return sweeps
    
def map_channels_to_cells(unique_channels,unique_cells, overview):
    channel_mapping={}
    for uncell in unique_cells:
        try:
            hs = str(int(overview[overview.cell == uncell].iloc[0].HS))
        except Exception as e:
            print(f'No input HS : {e}')
            hs='unknown'
        for element in unique_channels:
            if element[-1] == hs:
               channel_mapping[element] = uncell
    return channel_mapping


#get the multi sweeptable
def process_multi_sweep_table(dataset,rectype,s,cell,overview,ref=0):    
    df=overview[overview.cell == cell]
    beg_swp=df.fswp_condition.iloc[0]
    #create the sweeps table
    try:
        #identify the amount of channels in the file and correspond them into the cells using overview as a reference. 
        unique_channels = set(name.split('_')[-1] for name in dataset['acquisition'].keys())
        if len(unique_channels)!=1:
            print (f'multiple channels detected = {unique_channels}, using the channels from overview metadata to map the cells')
            _,unique_cells=n_cell_in_file(s)
            channel_mapping = map_channels_to_cells(unique_channels,unique_cells, overview)
        #get the sweeps data into the table
        sweeps=pd.DataFrame(list(dataset['acquisition'].keys()))
        sweeps = sweeps.rename(columns={0: 'key_name'})
        sweeps['sweep_number'] = sweeps['key_name'].apply(lambda x: int(x.split('_')[1]))
        sweeps['channel'] = sweeps['key_name'].apply(lambda x: x[-3:])
        #construct a sweep table
        disitiser = list(dataset['general']['labnotebook'].keys())[0]
        for i in range(0,len(dataset['general']['labnotebook'][disitiser]['textualValues'])):
            temp=dataset['general']['labnotebook'][disitiser]['textualValues'][i]
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
            timestamp=float(temp[2][0])
            readable_timestamp=convert_timestamp_to_human(timestamp)
            sweeps.at[sweeps['sweep_number'] == sweep_number, 'stimulus_code'] = stimulus_code[:-5]
            sweeps.at[sweeps['sweep_number'] == sweep_number, 'stimulus_unit'] = stimulus_unit
            sweeps.at[sweeps['sweep_number'] == sweep_number, 'clamp_mode'] = clamp_mode
            sweeps.at[sweeps['sweep_number'] == sweep_number, 'date'] = readable_timestamp.date()
            sweeps.at[sweeps['sweep_number'] == sweep_number, 'time'] = readable_timestamp.time().strftime('%H:%M')
            sweeps.at[sweeps['sweep_number'] == sweep_number, 'time_unix'] = timestamp
    except Exception as e:
            print(f'Could not load in the filtered sweep table for cell {cell} : {e}')        
    #fill the sweep table with data - BB, leak, compensation ; calculate Ra        
    for index, row in sweeps.iterrows():
       #register the holding and bridge balance
       key_name = row['key_name']
       indexx=key_name.split('_')[1]
       hs=key_name[-1]
       holding = np.nan
       bbalance=np.nan
       if 'bias_current' in dataset['acquisition'][key_name].keys():
           holding = dataset['acquisition'][key_name]['bias_current'][0]
       if 'bridge_balance' in dataset['acquisition'][key_name].keys():
           bbalance = dataset['acquisition'][key_name]['bridge_balance'][0]    
       sweeps.at[index, 'leak_pa']= holding 
       sweeps.at[index, 'bridge_balance_mohm']= bbalance 
       #locate capacitance compensation 
       if 'whole_cell_capacitance_comp' in dataset['acquisition'][f'data_{indexx}_AD{hs}'].keys(): 
            sweeps.at[index, 'compensation'] = 'yes'
       else:
            sweeps.at[index, 'compensation'] = 'no'
       #get the values to get Ra, Rm and tau 
       #get the sweeps data and attributes
       sweepi = dataset['acquisition'][key_name]['data']
       sweepv=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
       fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
       #get the testpulse timestamps (indexes)
       try:
            stimulus_changes = np.where(sweepv[:-1] != sweepv[1:])[0]
       except Exception as e:
            print(f'Could not find the TestPulse : {e}')
            tp_start, tp_end = None
            ra = np.nan
            sweeps.at[index, 'Ra'] = ra
       if len(stimulus_changes) < 2:
            sweeps.at[index, 'Ra'] = np.nan
            sweeps.at[index, 'compensation'] = 'no'
            print(f"No TP epoch found for sweep {row.sweep_number}")
            pass
       else:
            tp_start =stimulus_changes[0] + 1
            tp_end = stimulus_changes[1] + 1
       #define the input and output traces
       v=sweepv[: tp_end]
       i=sweepi[: tp_end]
       #calculate Ra and tau
       try:
            bsl = np.mean(i[tp_start-int(fs*0.002):tp_start-int(fs*0.001)])
            peak, peak_id = max(i, key=abs), np.argmax(np.abs(i))
            tp_inp= round(max(v, key=abs))
            if row.clamp_mode == "VoltageClamp":
                ra=tp_inp/(peak-bsl)*1000
                if ra > 0:
                    sweeps.at[index, 'Ra'] = ra
                else: 
                    sweeps.at[index, 'Ra'] = np.nan
                #calculate tau
                y=i[peak_id:]
                x = np.arange(0, len(y))
                m_guess = 500
                b_guess = -100
                t_guess = 0.03
                if len(y) !=0:
                    try:
                        (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess],maxfev=100000)
                        tau=-1/(-t_true)/fs*1000
                        sweeps.at[index, 'tau'] = tau
                    except Exception as e:
                        print(f'could not fit the tau for sweep {row.sweep_number} : {e}')
                        tau=np.nan
                        sweeps.at[index, 'tau'] = tau
         
            elif row.clamp_mode == "CurrentClamp":
                rm=(tp_inp/(peak-bsl))
                sweeps.at[index, 'Rm'] = rm
                
       except Exception as e:
            print(f'Could not calculate Rm and Ra for sweep {key_name} : {e}')
            sweeps.at[index, 'compensation'] = 'no'
            sweeps.at[index, 'Ra'] = np.nan
            sweeps.at[index, 'tau'] = np.nan
            sweeps.at[index, 'Rm'] = np.nan
       #allocate cells into corresponding rows, split the table in channels and
       if len(unique_channels)!=1:
           for channel in unique_channels:
               sweeps.at[sweeps['channel'] == channel, 'cell'] = channel_mapping.get(channel)
       else:         
               sweeps['cell'] = cell
       #split the table into separate cells, with a drop inindex for easier indexing
       all_sweeps = {}
       grouped = sweeps.groupby('cell')
       # Iterate over each group and save the corresponding table
       for key, group in grouped:
           all_sweeps[key] = group.reset_index(drop=True)
       #add noramlised Ra; 
       for key in all_sweeps.keys(): 
           #DEFINE IMPORTANT BASICS
            table=all_sweeps[key]
            table_metadata=overview[overview.cell == key]
            group = table_metadata.group.iloc[0]
            unique_conditions= table_metadata['condition'].unique()
            if 'vehicle' in unique_conditions:
                exptype='control'
            elif 'naag' in unique_conditions:
                exptype='experiment'
            table = table[table['sweep_number'] >= beg_swp]
            table['group'] = group
            table['exp_type'] = exptype
            #assign Ra_norm
            if 'Ra' in table.columns and not table['Ra'].dropna().empty:
                if len(table[table.clamp_mode == 'VoltageClamp']) != 0:
                    ra_norm_factor = table['Ra'].dropna().iloc[ref] #choose here you first reference Ra
                    table['Ra_norm'] = table['Ra'] / ra_norm_factor
                else:
                    print("No valid 'Ra' values found. Defaulting ra_norm_factor to 1.")
            else:
                ra = np.nan
                table.at[index, 'Ra'] = ra  
                table.at[index, 'Ra_norm'] = np.nan
            #add set_number and time for the graph of washin time
            for condition in unique_conditions: 
                start_condition = table_metadata[table_metadata['condition'] == condition]['fswp_condition'].iloc[0]
                table.at[table['sweep_number'] >= start_condition,'condition'] = condition
            #add washin state to cells which it is applicable to 
            for idx, row in table.iterrows():
                main, state = split_single_condition(row['condition'])
                table.at[idx, 'condition'] = main
                table.at[idx, 'wash_in_state'] = state
            #apply QC
            qckech = np.nan
            table['QC'] = qckech
            table.loc[(table['Ra_norm'] >= 0.7) & (table['Ra_norm'] <= 1.3), 'QC'] = 'included'
            table.loc[(table['Ra_norm'] < 0.7) | (table['Ra_norm'] > 1.3) | (table['Ra'] > 20), 'QC'] = 'excluded'
            table['QC'].fillna(method='ffill', inplace=True)
            table['QC'].fillna(method='bfill', inplace=True)
            #apply extended QC (QC change is too high, but Ra absolute is below 20)
            table=extend_qc_check(table)
            all_sweeps[key]=table
    
    return all_sweeps    


def assign_condition_sets(sweeps):
    # Create a copy to avoid modifying the original DataFrame
    sweeps = sweeps.copy()
    
    # Initialize set_number column
    if 'set_number' not in sweeps.columns:
        sweeps['set_number'] = np.nan
    # Safety check - make sure we have condition and stimulus_code columns
    if 'condition' not in sweeps.columns:
        print("Warning: 'condition' column missing. Adding empty column.")
        sweeps['condition'] = np.nan
        return sweeps
    if 'stimulus_code' not in sweeps.columns:
        print("Warning: 'stimulus_code' column missing. Adding empty column.")
        sweeps['stimulus_code'] = np.nan
        return sweeps
    # Fill NaN values in condition column with a placeholder to avoid issues
    sweeps['condition'] = sweeps['condition'].fillna('unknown')
    # Convert stimulus_code to string to safely use string methods
    sweeps['stimulus_code'] = sweeps['stimulus_code'].astype(str)
    # For each condition
    for condition in sweeps['condition'].unique():
        if condition == 'unknown':
            continue
        # Get data for this condition only
        condition_data = sweeps[sweeps['condition'] == condition]
        # Find A1_Test locations
        test_locs = condition_data[condition_data['stimulus_code'].str.contains('A1_Test', na=False)].index.tolist()
        # Initialize set counter for this condition
        set_counter = 1  
        # Check each A1_Test location
        for i, loc in enumerate(test_locs):
            # Get index position in the full DataFrame
            all_indices = sweeps.index.tolist()
            try:
                loc_position = all_indices.index(loc)
            except ValueError:
                print(f"Warning: Index {loc} not found in all_indices")
                continue
            # Check if there's a next index
            if loc_position + 1 < len(all_indices):
                next_loc = all_indices[loc_position + 1]
                next_sweep = sweeps.loc[next_loc, 'stimulus_code']
                if isinstance(next_sweep, str) and 'A2_CC' in next_sweep:
                    # Assign set number if A2_CC follows
                    sweeps.loc[loc, 'set_number'] = set_counter
                    set_counter += 1
    # Fill forward within each condition group
    for condition in sweeps['condition'].unique():
        if condition == 'unknown':
            continue
        mask = sweeps['condition'] == condition
        sweeps.loc[mask, 'set_number'] = sweeps.loc[mask, 'set_number'].fillna(method='ffill')
    # Revert 'unknown' back to NaN
    sweeps.loc[sweeps['condition'] == 'unknown', 'condition'] = np.nan
    return sweeps

def filter_the_signal(trace, fs, cutoff=2000, order=3):
    #nyq is half of the sampling rate
    nyq=fs/2
    #cutoff frequency is frequency above which the signal will be attenuated.
                    #for spike-containing data it is good 1-5 depending on how much there is. More noise - higher cutoff        
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_trace = filtfilt(b, a, trace)
    return filtered_trace


def analyse_CC_spikes(sweeps,dataset,cell,overview,rectype,protocol_str='teps',for_database=False):
    all_spikes=pd.DataFrame()   
    #select the correct protocol - CCsteps with clamping - analyse al the protocols there are
    df2 = sweeps[(sweeps.stimulus_code.str.contains('teps')) &
                 (sweeps.clamp_mode.values == 'CurrentClamp')]
    if for_database:
        df2 = sweeps[(sweeps.stimulus_code.str.contains(protocol_str)) &
                 (sweeps.clamp_mode.values == 'CurrentClamp') & (sweeps.condition == 'baseline') & ~(sweeps.leak_pa.isna())]
    #loop over sweeps to load them in and analyse
    if df2.shape[0] != 0 and (not for_database or len(df2.condition.unique()) > 0):
        for index, row in df2.iterrows():
            print(f'analysing sweep {row.sweep_number}')
            try:
                if rectype == 'mono':
                    sweep = dataset.sweep(row.sweep_number)
                    sweepi=sweep.i
                    sweepv=sweep._response
                    sweept=sweep.t
                    fs=sweep.sampling_rate
                    if sweep.epochs['stim'] is not None:
                        stim_start, stim_end = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                else:
                    key_name = row['key_name']
                    indexx=key_name.split('_')[1]
                    hs=key_name[-1]
                    sweepv = dataset['acquisition'][key_name]['data']
                    sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
                    fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
                    sweept=np.linspace(0, len(sweepv) / fs, len(sweepv), endpoint=False)
                    stimulus_changes = np.where(sweepi[:-1] != sweepi[1:])[0]
                    if len(stimulus_changes) !=2:
                        stim_start, stim_end = stimulus_changes[2], stimulus_changes[3]
                cmode = 'clamped' if not pd.isna(row.leak_pa) else 'not_clamped'
                exptype=df2.exp_type.iloc[0]
                curr_inj = round(max(sweepi[stim_start: stim_end],key=abs))
                #filter
                filt_v = filter_the_signal(sweepv, fs)
                if curr_inj > 0: 
                    ext = SpikeFeatureExtractor()
                    res = ext.process(sweept, filt_v, sweepi) #process the sweeps only in the curr injection, not before or after (discarding the rebound spikes)
                    if len(res)==0:  # If res is empty, initialize it as an empty DataFrame
                        res = pd.DataFrame({'cell': [cell],'group':[row.group],'sweep_number': [row.sweep_number],'condition': [row.condition],
                                            'QC':[row.QC],'QC_extended':[row.QC_extended],'QC_spikes': 'included', 'exp_type': [exptype], 'clamp_mode':cmode, 'curr_inj': [curr_inj]})
                        #set nAPs into sweeps for FI curve
                        sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr_inj
                        sweeps.at[sweeps['sweep_number'] == row.sweep_number,'nAPs'] = 0
                    elif len(res) > 0:
                        if any(res.threshold_i == 0):
                            res['QC_spikes'] = 'excluded'
                        else:
                            res['QC_spikes'] = 'included'
                        valid_spikes = res[(res['threshold_index'] >= stim_start) & (res['threshold_index'] <= stim_end)]
                        if len(valid_spikes)==0: 
                           res = pd.DataFrame({'cell': [cell],'group':[row.group],'sweep_number': [row.sweep_number],'condition': [row.condition],
                                            'QC':[row.QC],'QC_extended':[row.QC_extended],'QC_spikes': 'included', 'exp_type': [exptype], 'clamp_mode':cmode, 'curr_inj': [curr_inj]})
                           #set nAPs into sweeps for FI curve
                           sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr_inj
                           sweeps.at[sweeps['sweep_number'] == row.sweep_number,'nAPs'] = 0 
                        else:
                            res=valid_spikes
                            res['cell']=row.cell
                            res['group']=row.group
                            res['sweep_number']=row.sweep_number
                            #res['set_number']=row.set_number
                            res['condition']=row.condition
                            res['QC'] = row.QC
                            res['QC_extended'] = row.QC_extended
                            res['exp_type']= exptype
                            res['clamp_mode']=cmode
                            res['curr_inj']=curr_inj
                            #create a f_bin label per spike
                            res['ISI'] = res['threshold_t'].diff()
                            res['ISI'].fillna(0, inplace=True) #replace first np.nan with 0
                            res['IF']=1/res['ISI']
                            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
                            labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']
                            # Use pd.cut() to categorize the 'IF' values into bins
                            res['f_bin'] = pd.cut(res['IF'], bins=bins, labels=labels, right=False, include_lowest=True)
                            res['f_bin'] = res['f_bin'].cat.add_categories(['First AP'])
                            res.at[0,'f_bin'] = 'First AP' #make sure First AP of the train is categoried
                            #add broader Fbins
                            broad_mapping = {'First AP': 'First AP','0-10': '0-20','10-20': '0-20','20-30': '20-50','30-40': '20-50','40-50': '20-50'}
                            res['f_bin_broad'] = res['f_bin'].map(broad_mapping)
                            #set nAPs into sweeps for FI curve
                            sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr_inj
                            sweeps.at[sweeps['sweep_number'] == row.sweep_number,'nAPs'] = res['threshold_t'].notnull().sum()
                    all_spikes = pd.concat([all_spikes, res], axis = 0)
            except Exception as e:
                continue  # Skip this sweep if error occurs    
                print(f'{e}')
        print(f'total Steps were analysed for cell {cell}')
    else:
        print(f'cell {cell} does not contian data / paired data to analyse')
        return sweeps, None
    return sweeps, all_spikes

def analyse_CC_sag(sweeps,dataset,cell,overview, rectype, protocol_str='teps', for_database=False):
    #define the ouput table
    sag_result=pd.DataFrame(columns = ['cell','group', 'sweep_number','condition','QC','QC_extended', 'exp_type', 'set_number',
                                       'clamp_mode','curr_inj','baseline_v','sag_volt',
                                       'ss_volt', 'sag_deflection','ss_deflection','tau','R2',
                                       'sag_ratio'])

    #select the correct protocol - CCsteps with clamping - analyse al the protocols there are
    df2 = sweeps[(sweeps.stimulus_code.str.contains(protocol_str)) &
                 (sweeps.clamp_mode.values == 'CurrentClamp')]
    if for_database:
        df2 = sweeps[(sweeps.stimulus_code.str.contains(protocol_str)) &
                 (sweeps.clamp_mode.values == 'CurrentClamp') & (sweeps.condition == 'baseline')& ~(sweeps.leak_pa.isna()) ]
    #loop over sweeps to load them in and analyse
    if df2.shape[0] != 0 and (not for_database or len(df2.condition.unique()) > 0):
            for index, row in df2.iterrows():
                print(f'analysing sweep {row.sweep_number}')
                try:
                    if rectype == 'mono':
                        sweep = dataset.sweep(row.sweep_number)
                        sweepi=sweep.i
                        sweepv=sweep._response
                        fs=sweep.sampling_rate
                        if sweep.epochs['stim'] is not None:
                            stim_start, stim_end = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                    else:
                        key_name = row['key_name']
                        indexx=key_name.split('_')[1]
                        hs=key_name[-1]
                        sweepv = dataset['acquisition'][key_name]['data']
                        sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
                        fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
                        stimulus_changes = np.where(sweepi[:-1] != sweepi[1:])[0]
                        if len(stimulus_changes) !=2:
                            stim_start, stim_end = stimulus_changes[2], stimulus_changes[3]
                    cmode = 'clamped' if not pd.isna(row.leak_pa) else 'not_clamped'
                    exptype=df2.exp_type.iloc[0]
                    curr_inj = round(max(sweepi[stim_start: stim_end],key=abs))
                    #filter
                    filt_v = filter_the_signal(sweepv, fs)
                    #if sweep has negative input, analyse sag and calculate Rin - fit in the curve into input-output
                    if curr_inj < 0:
                       sag_ind = stim_start+np.argmin(filt_v[stim_start:stim_start+int(fs*0.2)]) #sag is searched in the first 200ms after curr_inj
                       baseline = np.mean(filt_v[stim_start-int(fs*0.1):stim_start], axis=0)
                       sagv= np.mean(filt_v[sag_ind-int(fs*0.001/2):sag_ind+int(fs*0.001/2)], axis=0) # sagv is a mean value of 1ms (+- 0.5ms) around the max min voltage
                       ss_volt = np.mean(filt_v[stim_end-int(fs*0.1):stim_end], axis=0) # sagv is a mean value of 1ms (+- 0.5ms) around the max min voltage 
                       sag_deflection = baseline - sagv
                       ss_deflection = baseline - ss_volt
                       sag_ratio = ss_deflection / sag_deflection
                       #calculation of tau fitted in the first 100ms as a fit to single exponential 
                       y=filt_v[stim_start:int(stim_start+(fs*0.1))]
                       x = np.arange(0, len(y))
                       m_guess = 10
                       b_guess = -100
                       t_guess = 0.03
                       if len(y) !=0:
                            try:
                                (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess],maxfev=100000 )
                                tau=-1/(-t_true)/fs*1000
                                y_pred = monoExp(x, m_true, t_true, b_true)
                                ss_res = np.sum((y - y_pred) ** 2)
                                ss_tot = np.sum((y - np.mean(y)) ** 2)
                                r2 = 1 - (ss_res / ss_tot)
                                if r2>0.95:
                                   tau=tau
                                else:
                                    tau=np.nan
                            except Exception as e:
                                 print(f'could not fit the tau for sweep {row.sweep_number} : {e}')
                                 tau=np.nan
                                 r2 = np.nan
                       sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr_inj
                       sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'nAPs'] = 0
                       sag_result= sag_result.append({'cell':row.cell, 'group':row.group,'sweep_number': row.sweep_number, 'condition': row.condition,'QC':row.QC,'QC_extended':row.QC_extended,'exp_type':exptype, 
                                                      'clamp_mode':cmode,'curr_inj':curr_inj, 'baseline_v':baseline,'sag_volt':sagv,
                                                      'ss_volt':ss_volt, 'sag_deflection':sag_deflection, 'ss_deflection':ss_deflection,
                                                       'tau':tau, 'R2': r2,'sag_ratio':sag_ratio}, ignore_index=True)
                    elif curr_inj == 0: 
                        sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'curr_inj'] = curr_inj
                        sweeps.at[sweeps['sweep_number'] == row.sweep_number,'nAPs'] = 0
                        continue
                except Exception as e:
                    continue  # Skip this sweep if error occurs
            print(f'total sags were analysed for cell {cell}')
    return sweeps,  sag_result

def assign_sets_after_analysis(sweeps, spike_results, sag_results, target_protocol='teps'):
    sweeps['set_number'] = np.nan
    spike_results['set_number'] = np.nan
    sag_results['set_number'] = np.nan
    # Group by condition and protocol
    protocol_mask = sweeps['stimulus_code'].str.contains(target_protocol)
    protocol_sweeps = sweeps[protocol_mask]
    for condition in protocol_sweeps['condition'].unique():
        condition_sweeps = protocol_sweeps[protocol_sweeps['condition'] == condition]
        condition_sweeps_sorted = condition_sweeps.sort_values('sweep_number')
        # Find break point by large jumps in curr_inj
        curr_inj_series = condition_sweeps_sorted['curr_inj']
        break_indices = np.where(np.abs(np.diff(curr_inj_series)) > 100)[0]
    
        if len(break_indices) == 0:
            # All in set 1 if no break
            sweeps.loc[condition_sweeps_sorted.index, 'set_number'] = 1
        else:
            # Split into sets
            start_index = 0
            for set_num, break_point in enumerate(break_indices, start=1):
                # Sweeps from start_index to break_point (inclusive)
                set_indices = condition_sweeps_sorted.index[start_index:break_point+1]
                sweeps.loc[set_indices, 'set_number'] = set_num
                start_index = break_point + 1
            # Handle any remaining sweeps after last break
            if start_index < len(condition_sweeps_sorted):
                final_set_indices = condition_sweeps_sorted.index[start_index:]
                sweeps.loc[final_set_indices, 'set_number'] = len(break_indices) + 1
            
        if spike_results is not None:
           spike_results['set_number'] = spike_results['sweep_number'].map(sweeps.loc[protocol_mask].set_index('sweep_number')['set_number'])
   
        if sag_results is not None:
           sag_results['set_number'] = sag_results['sweep_number'].map(sweeps.loc[protocol_mask].set_index('sweep_number')['set_number'])
    
    return sweeps, spike_results, sag_results

def get_fitted_features(sags,sweeps,ifolder, condition_palette=condition_palette, plot_figure=True):
    output = pd.DataFrame(columns=['cell', 'clamp_mode', 'set_number', 'condition', 'condition_set', 'Rin', 'ss_slope', 'sag_slope', 'ratio_of_slopes'])
    cell = sags.cell.iloc[0]
    
    unique_combos = sags[['condition', 'set_number']].drop_duplicates().reset_index(drop=True)

    n_combos = len(unique_combos)
    fig, axes = plt.subplots(n_combos, 3, figsize=(15, 5*n_combos))
    fig.suptitle(f"{cell} - fitted features", fontsize=16, y=0.98)
    # Flatten axes if only one combination
    if n_combos == 1:
        axes = axes.reshape(1, -1)
        
    for idx, row in unique_combos.iterrows():
        condition = row['condition']
        set_num = row['set_number']
        # Select data for this condition-set
        df_set = sags[(sags['condition'] == condition) & (sags['set_number'] == set_num)]
        df_set = df_set.dropna(subset=['curr_inj', 'sag_deflection', 'ss_volt', 'sag_volt'])
        clamping = df_set['clamp_mode'].iloc[0]
        if len(df_set) != 0:
            ax_rin = axes[idx, 1]
            # Calculate and plot Rin
            coeffs_rin = np.polyfit(list(df_set['curr_inj']), list(df_set['sag_deflection']), 1)
            rin_fitted = abs(coeffs_rin[0])*1000
            if plot_figure:
                ax_rin.scatter(df_set['curr_inj'], df_set['sag_deflection'], 
                            color=condition_palette[condition], 
                            label=f'{condition} Set {int(set_num)}')
                rin_poly_func = np.poly1d(coeffs_rin)
                x_range_rin = np.linspace(df_set['curr_inj'].min(), df_set['curr_inj'].max(), 100)
                ax_rin.plot(x_range_rin, rin_poly_func(x_range_rin), 
                        color='gray' if condition == 'baseline' else 'black', 
                        linestyle='--')
                ax_rin.text(0.05, 0.05, f'Rin {condition} Set {int(set_num)}: {rin_fitted:.2f}', 
                         transform=ax_rin.transAxes)
                ax_rin.set_xlabel('Current Injection')
                ax_rin.set_ylabel('sag deflection')
                
            # Calculate slopes
            ax_slopes = axes[idx, 0]
            ss_slope, ss_intercept = np.polyfit(list(df_set['curr_inj']), list(df_set['ss_volt']), 1)
            sag_slope, sag_intercept = np.polyfit(list(df_set['curr_inj']), list(df_set['sag_volt']), 1)
            ratio_of_slopes = sag_slope/ss_slope          
            if plot_figure:
                ax_slopes.scatter(df_set['curr_inj'], df_set['ss_volt'], 
                            color=condition_palette[condition], marker='o', 
                            label=f'{condition} Set {int(set_num)} SS')
                ax_slopes.scatter(df_set['curr_inj'], df_set['sag_volt'], 
                            color=condition_palette[condition], marker='o', 
                            facecolors='none', edgecolors=condition_palette[condition], 
                            label=f'{condition} Set {int(set_num)} Sag')
                
                ss_poly_func = np.poly1d([ss_slope, ss_intercept])
                sag_poly_func = np.poly1d([sag_slope, sag_intercept])
                x_range = np.linspace(df_set['curr_inj'].min(), df_set['curr_inj'].max(), 100)
                ax_slopes.plot(x_range, ss_poly_func(x_range), 
                        color='gray' if condition == 'baseline' else 'black', 
                        linestyle='--')
                ax_slopes.plot(x_range, sag_poly_func(x_range), 
                        color='gray' if condition == 'baseline' else 'black', 
                        linestyle=':')
                ax_slopes.text(0.95, 0.05, f'Ratio of slopes {condition} Set {int(set_num)}: {ratio_of_slopes:.2f}', 
                         transform=ax_slopes.transAxes, ha='right', va='bottom')
                ax_slopes.set_xlabel('Current Injection')
                ax_slopes.set_ylabel('Voltage')
                ax_slopes.legend(loc='upper left')
            
            #get rheobase and FI slope values
            ax_fi = axes[idx, 2]
            fi_set = sweeps[(sweeps['condition'] == condition) & (sweeps['set_number'] == set_num) & (sweeps['QC_spikes'] == 'included')]
            rheobase = np.nan
            fi_slope = np.nan
            has_spikes = False
            if len(fi_set) > 0 and 'nAPs' in fi_set.columns:
                if (fi_set.nAPs > 0).any():
                    has_spikes = True
                    rheobase = fi_set[fi_set.nAPs > 0]['curr_inj'].min()
                    # Calculate FI slope
                    fi_fit = fi_set[fi_set.curr_inj <= 250]
                    if len(fi_fit) >= 3:
                        fi_slope, fi_intercept = np.polyfit(list(fi_fit['curr_inj']), list(fi_fit['nAPs']), 1)
    
            #plot
            if plot_figure:
                if has_spikes:
                    ax_fi.scatter(fi_set['curr_inj'], fi_set['nAPs'], 
                                    color=condition_palette[condition])
                    fi_poly_func = np.poly1d([fi_slope, fi_intercept])
                    x_range_fi = np.linspace(fi_fit['curr_inj'].min(), fi_fit['curr_inj'].max(), 100)
                    ax_fi.plot(x_range_fi, fi_poly_func(x_range_fi), 
                            color='gray' if condition == 'baseline' else 'black',  linestyle='--')
                    if not np.isnan(rheobase):
                        ax_fi.text(0.05, 0.95, f'FI slope {condition} Set {int(set_num)}: {fi_slope:.2f}\n Rheobase {int(rheobase)}', 
                             transform=ax_fi.transAxes,  ha='left', va='top')
                    else:
                        ax_fi.text(0.05, 0.95, f'FI slope {condition} Set {int(set_num)}: {fi_slope:.2f}', 
                             transform=ax_fi.transAxes,  ha='left', va='top')
                    ax_fi.set_xlabel('Current Injection')
                    ax_fi.set_ylabel('number of APs')
                else:
                    ax_fi.set_visible(False)
           
            # Add to output
            new_row = pd.DataFrame({
                'cell': [cell],
                'clamp_mode': [clamping],
                'set_number': [set_num],
                'condition': [condition],
                'Rin': [rin_fitted],
                'ss_slope': [ss_slope],
                'sag_slope': [sag_slope],
                'ratio_of_slopes': [ratio_of_slopes], 
                'rheobase': [rheobase], 
                'fi_slope': [fi_slope]})
            output = pd.concat([output, new_row])
            
        plt.tight_layout()   
        plt.savefig(f'{ifolder}/{cell}_fitted_features.eps')
    
    return output

def create_excitability_summary_table(cell_spikes):
    # Create empty dataframe for results
    results = pd.DataFrame(columns=['cell','condition','clamp_mode', 'set_number', 'f_bin',
                                    'threshold_i','threshold_v', 'threshold_t',
                                    'peak_i','peak_v', 'peak_t', 
                                    'upstroke', 'downstroke', 'width', 'upstroke_downstroke_ratio'])
    cell=cell_spikes.cell.iloc[0]
    unique_bins = cell_spikes['f_bin_broad'].dropna().unique()
    # Group by condition and set_number
    for (condition, set_num), group_df in cell_spikes[cell_spikes.QC_spikes == 'included'].groupby(['condition', 'set_number']):
        # Skip if no spikes
        if group_df['threshold_v'].isna().all():
            continue
       
        # Extract properties of first AP (rheobase AP)
        ap_data = group_df[group_df['threshold_v'].notna()]
        min_curr_inj = ap_data['curr_inj'].min()
        rheobase_aps = ap_data[ap_data['curr_inj'] == min_curr_inj]
        rheobase_ap=rheobase_aps.iloc[0]
        clamping = group_df['clamp_mode'].iloc[0]
        rheo_row = {'cell': cell,'condition': condition,'clamp_mode':clamping,'set_number': set_num,'f_bin': 'Rheobase AP',
                'threshold_i': rheobase_ap['threshold_i'],'threshold_v': rheobase_ap['threshold_v'],'threshold_t': rheobase_ap['threshold_t'],
                'peak_i': rheobase_ap['peak_i'],'peak_v': rheobase_ap['peak_v'],'peak_t': rheobase_ap['peak_t'],
                'upstroke': rheobase_ap['upstroke'],'downstroke': rheobase_ap['downstroke'],'width': rheobase_ap['width'],'upstroke_downstroke_ratio': rheobase_ap['upstroke_downstroke_ratio']}
        results = pd.concat([results, pd.DataFrame([rheo_row])], ignore_index=True)
        
        for bin_name in unique_bins:
            bin_data = group_df[group_df['f_bin_broad'] == bin_name]
            if len(bin_data) > 0:
                # Calculate mean AP properties for this bin
                bin_row = {'cell': cell, 'condition': condition,'clamp_mode':clamping,'set_number': set_num,'f_bin': bin_name,
                    'threshold_i': bin_data['threshold_i'].mean(),'threshold_v': bin_data['threshold_v'].mean(),'threshold_t': bin_data['threshold_t'].mean(),
                    'peak_i': bin_data['peak_i'].mean(),'peak_v': bin_data['peak_v'].mean(),'peak_t': bin_data['peak_t'].mean(),
                    'upstroke': bin_data['upstroke'].mean(), 'downstroke': bin_data['downstroke'].mean(),'width': bin_data['width'].mean(),'upstroke_downstroke_ratio': bin_data['upstroke_downstroke_ratio'].mean()}
                
                # Add to results
                results = pd.concat([results, pd.DataFrame([bin_row])], ignore_index=True)
    
    return results


def calculate_rmp_change(sweeps):   
    # Create empty dataframe for results
    results = pd.DataFrame(columns=['cell','group','exp_type', 'QC', 'QC_extended','sweep_number', 'protocol','time_unix',  'condition', 'rmp_mean'])
    # Filter for current clamp sweeps with no leak_pa value
    cc_sweeps = sweeps[(sweeps['clamp_mode'] == 'CurrentClamp') & (sweeps['leak_pa'].isna())]
    frstswp=sweeps.iloc[0]
    new_row = pd.DataFrame({'cell': [frstswp.cell],'group':[frstswp.group],'QC':[frstswp.QC], 'QC_extended':[frstswp.QC_extended],
                            'sweep_number': [frstswp.sweep_number],'protocol':[frstswp.stimulus_code],'time_unix': [frstswp.time_unix],'condition': [frstswp.condition],'rmp_mean': [np.nan]})
    results = pd.concat([results, new_row], ignore_index=True)
    # Process each sweep
    for index, row in cc_sweeps.iterrows():
        cell = row['cell']
        sweep_number = row['sweep_number']
        condition = row['condition']
        protocol=row['stimulus_code']
        time_unix=row['time_unix']   
        if rectype == 'mono':
            sweep=dataset.sweep(sweep_number)
            sweepi=sweep.i
            sweepv=sweep.v
            if 'test' and 'stim' in sweep.epochs.keys():
                if not sweep.epochs['stim'] is None:
                    stard_index, end_index= sweep.epochs['test'][-1], sweep.epochs['stim'][0]
                else: 
                    stard_index, end_index= sweep.epochs['test'][-1], sweep.epochs['recording'][-1]
            else: 
                print ('FIX THE EPOCH DETECTION for rmp measure')
            
            rmp_trace = sweepv[stard_index:end_index]
            
        else:
            key_name= row['key_name']
            indexx=key_name.split('_')[1]
            hs=key_name[-1]
            sweepv = dataset['acquisition'][key_name]['data']
            sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
            sweepi=np.array(sweepi)
            end_index=sweepi[np.where(~np.isnan(np.array(sweepi)))]
            stard_index, end_index= np.where(sweepi !=0)[0]
            sweepi
            rmp_trace=np.mean
        # Calculate RMP as mean voltage in baseline period
        
        rmp_mean = np.mean(rmp_trace)
        
        # Add to results
        new_row = pd.DataFrame({
            'cell': [cell],
            'group':[row.group],
            'exp_type': [row.exp_type], 
            'QC':[row.QC],
            'QC_extended':[row.QC_extended], 
            'sweep_number': [sweep_number],
            'protocol':[protocol],
            'time_unix': [time_unix],
            'condition': [condition],
            'rmp_mean': [rmp_mean]})
        
        results = pd.concat([results, new_row], ignore_index=True)
    
    results['time_diff']=results['time_unix']-results['time_unix'].iloc[0]
    
    return results


# def analyse_ps_protocols(sweeps):
#     for protocol in sweeps.stimulus_code.unique():
#         if 'SubT' in protocol:
#             _, subthresh=analyse_CC_sag(sweeps,dataset,cell,overview, rectype, protocol_str='SubT')
            
   
#     output=pd.DataFrame(columns=[])
#     return output
    
def downsample(dataset,sweep_nr,rectype, sweeps, down_rate=25000.0,mode='v'):
    if rectype == 'mono':
        fs=dataset.sweep(sweep_number=sweep_nr).sampling_rate
        if mode =='v':
            tr=dataset.sweep(sweep_nr).v
        elif mode == 'i':
            tr=dataset.sweep(sweep_nr).i
        elif mode=='t':
            tr=dataset.sweep(sweep_nr).t
            
        if fs == down_rate:
            return tr
    
        factor = int(fs /down_rate)
        if factor > 1:
            output = tr[::factor]
            return output
        else:
            raise ValueError("Sampling frequency is too low")
    else: 
        key_name = sweeps[sweeps.sweep_number==sweep_nr]['key_name'].iloc[0]
        indexx=key_name.split('_')[1]
        hs=key_name[-1]
        if mode =='v':
            tr= dataset['acquisition'][key_name]['data']
        elif mode == 'i':
            tr=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
        elif mode=='t':
            sweepv = dataset['acquisition'][key_name]['data']
            fs=int(1/sweepv.attrs['IGORWaveScaling'][1][0]*1000)
            tr=np.linspace(0, len(sweepv) / fs, len(sweepv), endpoint=False)
        sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
        fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
        if fs == down_rate:
            return tr
        factor = int(fs /down_rate)
        if factor > 1:
            output = tr[::factor]
            return output
        else:
            raise ValueError("Sampling frequency is too low")
            
            
def res_freq(dataset, sweeps, rectype, cell):
    #get the data from the dataset file
    fig, axes = plt.subplots(1, 3, figsize=(15,10))
    fig.suptitle(f"{cell} - CHIRP features", fontsize=16, y=0.98)
    if rectype == 'mono':
        df = sweeps[(sweeps.stimulus_code.str.contains('CHIRP'))&(sweeps.condition.isin(['baseline', 'naag', 'vehicle']))].reset_index(drop=True)
    else:
        if 'key_name' not in sweeps.columns or sweeps.cell.iloc[0] != cell:
            all_sweeps_from_file = process_multi_sweep_table(dataset,rectype,s,cell,overview,ref=0)
            sweeps=all_sweeps_from_file[cell]
        df = sweeps[(sweeps.stimulus_code.str.contains('CHIRP'))&(sweeps.condition.isin(['baseline', 'naag', 'vehicle']))].reset_index(drop=True)
        
        
    output=pd.DataFrame(columns = ['cell','condition','max_impedance', 'f_res', '3db_cutoff'])
    
    data = {cell: {}}
    available_conditions = df.condition.unique()
    for condition in available_conditions:
        data[cell][condition] = {'avg_trace': None,'impedance': None,'norm_impedance': None}


    if df.shape[0] != 0:
        condition_chirps = {cond: [] for cond in available_conditions}
        stim_traces=[]
        time_traces=[]
        min_length = None 
        
        for index, row in df.iterrows():
            if rectype == 'mono':
                sweep=dataset.sweep(sweep_number=row.sweep_number)
                v=sweep.v
                i=sweep.i
                t=sweep.t
                fs=sweep.sampling_rate
                
            else:
                key_name = row['key_name']
                indexx=key_name.split('_')[1]
                hs=key_name[-1]
                v = dataset['acquisition'][key_name]['data']
                i=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
                fs=int(1/i.attrs['IGORWaveScaling'][1][0]*1000)
                t=np.linspace(0, len(v) / fs, len(v), endpoint=False)
                valid_indices = ~np.isnan(v)
                v = v[valid_indices]
                i = i[:len(v)] if len(i) > len(v) else i[valid_indices[:len(i)]]
                t = t[:len(v)]
                
            condition=row.condition
            #get all the sweeps per condition to get the average
            if fs != 25000: #make sure they are all samples at 25000 (ipfx does 2000, I do 25000 + filtering at lowpass40: https://github.com/AllenInstitute/ipfx/blob/master/ipfx/chirp.py)
                v=downsample(dataset,row.sweep_number,rectype, sweeps,down_rate=25000)
                i=downsample(dataset,row.sweep_number,rectype, sweeps,down_rate=25000, mode='i')
                t=downsample(dataset,row.sweep_number,rectype, sweeps,down_rate=25000, mode='t')
            
            #remove nans    
            valid_indices = ~np.isnan(v)
            v = v[valid_indices]
            i = i[:len(v)] if len(i) > len(v) else i[valid_indices[:len(i)]]
            t = t[:len(v)]
            
            #make sure the sweep contains entire stimulation
            if max(t) !=  23.59996 and max(t)>20.6:
                if min_length is None or len(v) < min_length:
                    min_length=len(v)
            elif max(t) <  20.6: #as in ipfx (time until the recrodinggoes)
                continue
            
            if max(v)>-45: 
                continue
            
            #filter the trace
            v=filter_the_signal(v, 25000,1000,4)
            
            condition_chirps[condition].append(v)
            stim_traces.append(i)
            time_traces.append(t)
 
        traces = {}
        #process the averaged data
        if min_length is not None:
            for condition in available_conditions:
                adjusted_chirps = [arr[:min_length] for arr in condition_chirps[condition]]
                avg_trace = np.mean(adjusted_chirps, axis=0)
                data[cell][condition]['avg_trace'] = avg_trace
                traces[condition] = avg_trace
            
            stim_traces = [arr[:min_length] for arr in stim_traces]
            avg_stim = np.mean(stim_traces, axis=0)
            
            time_traces = [arr[:min_length] for arr in time_traces]
            avg_time = np.mean(time_traces, axis=0)
            
        else:
            for condition in available_conditions:
                avg_trace = np.mean(condition_chirps[condition], axis=0)
                data[cell][condition]['avg_trace'] = avg_trace
                traces[condition] = avg_trace
                
            avg_stim = np.mean(stim_traces, axis=0)     
            avg_time = np.mean(time_traces, axis=0)        
        
        stim = avg_stim
        t=avg_time


        plot_index = 1
        for condition in available_conditions:
            if plot_index <= 2:  # We only have 2 subplots for traces
                axes[plot_index].plot(t, data[cell][condition]['avg_trace'], color=condition_palette.get(condition, 'black'))
                axes[plot_index].set_xlabel('Time (s)', fontsize=12)
                axes[plot_index].set_ylabel('Voltage (mV)', fontsize=12)
                axes[plot_index].set_title(f'{cell} {condition} trace')
                plot_index += 1
        
        # If only one condition, adjust the unused plot
        if len(available_conditions) == 1:
            axes[2].set_visible(False)
        
        # #plot the outcome
        # axes[1].plot(t, data[cell]['baseline']['avg_trace'], color=condition_palette['baseline'])
        # axes[1].set_xlabel('Time (s)', fontsize=12)
        # axes[1].set_ylabel('Voltage (mV)', fontsize=12)
        # axes[1].set_title(f'{cell} baseline trace')
        # axes[2].plot(t, data[cell]['naag']['avg_trace'], color=condition_palette['naag'])
        # axes[2].set_xlabel('Time (s)', fontsize=12)
        # axes[2].set_ylabel('Voltage (mV)', fontsize=12)
        # axes[2].set_title(f'{cell} naag trace')
        
        start_index =np.where(np.abs(avg_time - 0.6) == np.min(np.abs(avg_time - 0.6)))[0][0]
        end_index = np.where(np.abs(avg_time - 20.6) == np.min(np.abs(avg_time - 20.6)))[0][0]
        
        # start_index, end_index = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
        
        for k in traces.keys():
            signal=traces[k][start_index:end_index]
            #plot the data
            L = len(signal)
            if rectype == 'mono':
                fs = sweep.sampling_rate
            else:
                i=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
                fs=int(1/i.attrs['IGORWaveScaling'][1][0]*1000)
            if fs != 25000.0:
                fs=25000.0
            f = np.arange(0, L) * fs / L
   
            #fast Fourier transformation
            signal_fft = np.fft.fft(signal)
            stim_fft = np.fft.fft(avg_stim[start_index:end_index])
            y=abs(signal_fft) / abs(stim_fft) * 1000
            #look only at 0.7-20 Hz
            xmin = 0.7
            xmax = 20
            mask = (f >= xmin) & (f <= xmax)
            zoomed_f = f[mask]
            zoomed_y = y[mask]
            cutoff_freq = 500
            normalized_cutoff_freq = cutoff_freq / (fs / 2)  # Normalized cutoff frequency
            b, a = butter(4, normalized_cutoff_freq, btype='lowpass', analog=False)
            fzoomed_y = filtfilt(b, a, zoomed_y)
           #smoothed_result = lowess(transfer_function, x, frac=smooth_fraction, return_sorted=False) # same function as eline uses to smooth
            #calculate empedance, 3db_cutoff, f_resonance
            max_imp=max(fzoomed_y)
            normalized = fzoomed_y/max_imp
            f_res = zoomed_f[np.argmax(fzoomed_y)]
            cutoff_3db=zoomed_f[np.where(normalized >= np.sqrt(0.5))[0][-1]]
            
            
            #plot the outcome
            axes[0].plot(zoomed_f, normalized)
            axes[0].axhline(y=np.sqrt(0.5), linewidth=2, linestyle=':', color='k')
            axes[0].axvline(x=f_res, color='red', linestyle='--')
            axes[0].set_ylabel('Impedance (normalized)', fontsize=12)
            axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
            axes[0].set_title(f'{cell} impedance')
            
            color = condition_palette.get(k, 'black')
            axes[0].plot(zoomed_f, normalized, label=k, color=color)
            axes[0].axhline(y=np.sqrt(0.5), linewidth=2, linestyle=':', color='k')
            axes[0].axvline(x=f_res, color='red', linestyle='--')
            
            output=output.append({'cell':cell,'condition':k, 'max_impedance': max_imp,'f_res':f_res, '3db_cutoff': cutoff_3db}, ignore_index=True)
            data[cell][k]['impedance'] = fzoomed_y
            data[cell][k]['zoomed_f'] = zoomed_f
            data[cell][k]['norm_impedance'] = normalized
        
        axes[0].set_ylabel('Impedance (normalized)', fontsize=12)
        axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[0].set_title(f'{cell} impedance')     
    plt.savefig(f'{ifolder}/{cell}_fres.eps')        
    return output, data

def res_freq_sets(dataset, sweeps, rectype, cell):
    # Prepare filtering for CHIRP stimuli
    if rectype == 'mono':
        df = sweeps[(sweeps.stimulus_code.str.contains('CHIRP'))].reset_index(drop=True)
    else:
        if 'key_name' not in sweeps.columns or sweeps.cell.iloc[0] != cell:
            all_sweeps_from_file = process_multi_sweep_table(dataset, rectype, s, cell, overview, ref=0)
            sweeps = all_sweeps_from_file[cell]
        df = sweeps[(sweeps.stimulus_code.str.contains('CHIRP'))].reset_index(drop=True)
    
    # Initialize output and data storage
    output = pd.DataFrame(columns=['cell', 'condition', 'CHIRP_set_number', 'max_impedance', 'f_res', '3db_cutoff'])
    data = {cell: {}}
    
    # Identify set breaks based on sweep number gaps
    # Ensure each set is from a single condition
    set_info = []
    current_set = 0
    current_condition = df.iloc[0]['condition']
    
    for i in range(len(df)):
        if (i > 0 and 
            (np.abs(df.iloc[i]['sweep_number'] - df.iloc[i-1]['sweep_number']) > 2 or 
             df.iloc[i]['condition'] != current_condition)):
            current_set += 1
            current_condition = df.iloc[i]['condition']
        
        set_info.append({
            'CHIRP_set_number': current_set,
            'condition': current_condition})
    
    # Add set information to dataframe
    df['CHIRP_set_number'] = [info['CHIRP_set_number'] for info in set_info]
    
    # Prepare figure for sets
    num_sets = df['CHIRP_set_number'].max() + 1
    num_rows = (num_sets + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4*num_rows))
    fig.suptitle(f"{cell} - CHIRP features by Sets", fontsize=16, y=0.98)
    
    # Ensure axes is 2D
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create impedance plot for all sets
    impedance_fig, impedance_ax = plt.subplots(figsize=(10, 6))
    impedance_fig.suptitle(f"{cell} - Impedance Profiles", fontsize=16)
    
    # Process each unique set
    for set_num in df['CHIRP_set_number'].unique():
        # Filter dataframe for this specific set
        set_df = df[df['CHIRP_set_number'] == set_num]
        # Collect traces for this set
        set_traces = []
        set_stim_traces = []
        set_time_traces = []
        min_length = None
        
        for index, row in set_df.iterrows():
            # Load sweep data
            if rectype == 'mono':
                sweep = dataset.sweep(sweep_number=row.sweep_number)
                v = sweep.v
                i = sweep.i
                t = sweep.t
                fs = sweep.sampling_rate
            else:
                key_name = row['key_name']
                indexx = key_name.split('_')[1]
                hs = key_name[-1]
                v = dataset['acquisition'][key_name]['data']
                i = dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
                fs = int(1/i.attrs['IGORWaveScaling'][1][0]*1000)
                t = np.linspace(0, len(v) / fs, len(v), endpoint=False)
            
            # Downsample to 25000 Hz
            if fs != 25000:
                v = downsample(dataset, row.sweep_number, rectype, sweeps, down_rate=25000)
                i = downsample(dataset, row.sweep_number, rectype, sweeps, down_rate=25000, mode='i')
                t = downsample(dataset, row.sweep_number, rectype, sweeps, down_rate=25000, mode='t')
            
            # Skip incomplete sweeps or sweeps with high voltage
            if max(t) != 23.59996 and max(t) > 20.6:
                if min_length is None or len(v) < min_length:
                    min_length = len(v)
            elif max(t) < 20.6:
                continue
            
            if max(v) > -45:
                continue
            
            # Filter the trace
            v = filter_the_signal(v, 25000, 1000, 4)
            
            set_traces.append(v)
            set_stim_traces.append(i)
            set_time_traces.append(t)
        
        # Adjust traces to minimum length if needed
        if min_length is not None:
            set_traces = [arr[:min_length] for arr in set_traces]
            set_stim_traces = [arr[:min_length] for arr in set_stim_traces]
            set_time_traces = [arr[:min_length] for arr in set_time_traces]
        
        # Average traces for this set
        avg_trace = np.mean(set_traces, axis=0)
        avg_stim = np.mean(set_stim_traces, axis=0)
        avg_time = np.mean(set_time_traces, axis=0)
        
        # Condition for this set (assuming all traces in a set have the same condition)
        condition = set_df['condition'].iloc[0]
        
        # Start and end indices for analysis
        start_index = np.where(np.abs(avg_time - 0.6) == np.min(np.abs(avg_time - 0.6)))[0][0]
        end_index = np.where(np.abs(avg_time - 20.6) == np.min(np.abs(avg_time - 20.6)))[0][0]
        
        signal = avg_trace[start_index:end_index]
        L = len(signal)
        
        # Frequency analysis
        f = np.arange(0, L) * 25000.0 / L
        signal_fft = np.fft.fft(signal)
        stim_fft = np.fft.fft(avg_stim[start_index:end_index])
        y = abs(signal_fft) / abs(stim_fft) * 1000
        
        # Look only at 0.7-20 Hz
        xmin, xmax = 0.7, 20
        mask = (f >= xmin) & (f <= xmax)
        zoomed_f = f[mask]
        zoomed_y = y[mask]
        
        # Filtering and impedance calculation
        cutoff_freq = 500
        normalized_cutoff_freq = cutoff_freq / (25000 / 2)
        b, a = butter(4, normalized_cutoff_freq, btype='lowpass', analog=False)
        fzoomed_y = filtfilt(b, a, zoomed_y)
        
        max_imp = max(fzoomed_y)
        normalized = fzoomed_y / max_imp
        f_res = zoomed_f[np.argmax(fzoomed_y)]
        cutoff_3db = zoomed_f[np.where(normalized >= np.sqrt(0.5))[0][-1]]
        
        # Plot traces in subplots
        row = set_num // 2
        col = set_num % 2
        axes[row, col].plot(avg_time, avg_trace, color=condition_palette[condition])
        axes[row, col].set_xlabel('Time (s)', fontsize=12)
        axes[row, col].set_ylabel('Voltage (mV)', fontsize=12)
        axes[row, col].set_title(f'Set {set_num} Trace ({condition})')
        
        # Plot impedance profile
        color = condition_palette.get(condition, 'gray')
        impedance_ax.plot(zoomed_f, normalized, color=color, label=f'Set {set_num} ({condition})')
        
        # Append to output
        output = output.append({
            'cell': cell,
            'condition': condition,
            'CHIRP_set_number': set_num,
            'max_impedance': max_imp,
            'f_res': f_res,
            '3db_cutoff': cutoff_3db
        }, ignore_index=True)
    
    # Finalize impedance plot
    impedance_ax.axhline(y=np.sqrt(0.5), linewidth=2, linestyle=':', color='k')
    impedance_ax.set_xlabel('Frequency (Hz)', fontsize=12)
    impedance_ax.set_ylabel('Impedance (normalized)', fontsize=12)
    impedance_ax.set_title(f'{cell} - Impedance Profiles')
    impedance_ax.legend()
    
    # Save figures
    plt.figure(fig.number)  # Make the traces figure the active figure
    plt.tight_layout()
    plt.savefig(f'{ifolder}/{cell}_fres_sets.eps')
    
    # Now switch to the impedance figure
    plt.figure(impedance_fig.number)
    plt.tight_layout()
    impedance_fig.savefig(f'{ifolder}/{cell}_impedance_profiles_sets.eps')
    
    return output, data


def cont_rmp(sweeps,dataset):
    output=pd.DataFrame(columns=['cell', 'group', 'exp_type','QC', 'QC_extended', 'sweep_number','protocol',  'condition', 'time_unix', 'time_bin', 'rmp_mean'])
    frstswp=sweeps.iloc[0]
    new_row = pd.DataFrame({'cell': [frstswp.cell],'group':[frstswp.group],'exp_type' : [frstswp.exp_type], 'QC':[frstswp.QC], 'QC_extended':[frstswp.QC_extended],
                            'sweep_number': [frstswp.sweep_number],'protocol':[frstswp.stimulus_code],'time_unix': [frstswp.time_unix],'condition': [frstswp.condition],'rmp_mean': [np.nan]})
    output = pd.concat([output, new_row], ignore_index=True)
    df=sweeps[(sweeps.clamp_mode == 'CurrentClamp')& (sweeps.leak_pa.isna())]
    
    for index,row in df.iterrows():
        swp=row['sweep_number']
        print(f'analysing sweep {swp}')
        protocol=row['stimulus_code']
        if re.search(r'Thr|Sear|Rhe|CHI|resp_|X5_|X6_|X7|X8|X9|Stim|baa|_5|_10|_20|_100', protocol):
            continue
        
        #load in the sweep data
        if rectype == 'mono':
            sweep=dataset.sweep(swp)
            sweepv=sweep.v
            sweepi=sweep.i
            sweept=sweep.t
            fs=sweep.sampling_rate
        else:
            key_name= row['key_name']
            indexx=key_name.split('_')[1]
            hs=key_name[-1]
            sweepv = dataset['acquisition'][key_name]['data']
            sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
            fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
            sweept=np.linspace(0, len(sweepv) / fs, len(sweepv), endpoint=False)
        print(f'data loaded sweep {swp}')
        #downsample to kHz
        if fs != 10000:
           print(f'downsampling sweep {swp}')
           v=downsample(dataset,swp,rectype, sweeps,down_rate=10000) 
           t=downsample(dataset,swp,rectype, sweeps,down_rate=10000,mode='t') 
           i=downsample(dataset,swp,rectype, sweeps,down_rate=10000,mode='i') 
           down_factor=fs/10000
           print(f'downsampled sweep {swp}')
        else: 
            v=sweepv
            t=sweept
            i=sweepi
        
        #check for invalid nan traces
        valid_indices = ~np.isnan(v)
        v = v[valid_indices]
        i = i[:len(v)] if len(i) > len(v) else i[valid_indices[:len(i)]]
        t = t[:len(v)]
        print(f'took out nans sweep {swp}')
        if len(v) == 0:
            continue
        
        #process the way calculate rmp does it. 
        if re.search(r'teps', protocol):
            if rectype == 'mono':
               if 'test' and 'stim' in sweep.epochs.keys():
                    if not sweep.epochs['stim'] is None:
                        stard_index, end_index= int(sweep.epochs['test'][-1]/down_factor), int(sweep.epochs['stim'][0]/down_factor)
                    else: 
                        stard_index, end_index= int(sweep.epochs['test'][-1]/down_factor), int(sweep.epochs['recording'][-1]/down_factor)
               else: 
                   print ('FIX THE EPOCH DETECTION for rmp measure')
                    
               v_filtered = v[stard_index:end_index]
               try:
                   v_filtered = filter_the_signal(v_filtered, fs, cutoff=50)
               except:
                   v_filtered=v_filtered
               rmp_mean=np.mean(v_filtered)
               
               output_row = {'cell': row['cell'], 'group': row['group'], 'exp_type': row['exp_type'],
                    'QC': row['QC'], 'QC_extended': row['QC_extended'],'sweep_number': swp,'protocol':row['stimulus_code'],  'condition': row['condition'], 
                    'time_unix': row['time_unix'] , 'time_bin': 0, 'rmp_mean': rmp_mean}
                
               output = output.append(output_row, ignore_index=True)
            else:
                i=np.array(i)
                end_index=i[np.where(~np.isnan(np.array(i)))]
                changeindexes=np.where(np.abs(np.diff(i)) > 1)[0]
                if len(changeindexes)==2:
                    stard_index, end_index= changeindexes[-1], len(i)
                else:
                    stard_index, end_index= changeindexes[1], changeindexes[2]
                v_filtered = v[stard_index:end_index]
                try:
                    v_filtered = filter_the_signal(v_filtered, fs, cutoff=50)
                except:
                    v_filtered=v_filtered
                rmp_mean=np.mean(v_filtered)
                
                output_row = {'cell': row['cell'], 'group': row['group'], 'exp_type': row['exp_type'],
                    'QC': row['QC'], 'QC_extended': row['QC_extended'],'sweep_number': swp,'protocol':row['stimulus_code'],  'condition': row['condition'], 
                    'time_unix': row['time_unix'] , 'time_bin': 0, 'rmp_mean': rmp_mean}
                
                output = output.append(output_row, ignore_index=True)
            continue
            
              
        if any(i>0):
            print(f'removing not zero stimulus sweep {swp}')
            indexes=np.where(i>0)[0]
            index_changes=indexes[np.where(np.abs(np.diff(indexes)) > 1)[0]]
            additional_indexes = []
            for change_index in index_changes:
                extrai = change_index + 1000
                additional_indexes.extend(range(change_index + 1, extrai + 1))
            indexes = sorted(list(indexes) + additional_indexes)
            v_filtered = v[~np.isin(np.arange(len(v)), indexes)]
            try:
                v_filtered = filter_the_signal(v_filtered, fs, cutoff=50)
            except: 
                v_filtered=v_filtered
            t_filtered = t[~np.isin(np.arange(len(t)), indexes)]
            i_filtered = i[~np.isin(np.arange(len(i)), indexes)]
        else: 
            try:
                v_filtered = filter_the_signal(v, fs, cutoff=50)
            except: 
                v_filtered=v_filtered
            t_filtered = t
            i_filtered = i
            
            
        #exclude the testpulse data - start the analysis from after the TP only
        print(f'removing TP sweep {swp}')
        if len(i_filtered)>0:
            start_index=((np.where(i_filtered[:-1] != i_filtered[1:])[0]) + 1)[-1]
        else:
            t_filtered=np.linspace(0, len(v_filtered) / 10000, len(v_filtered), endpoint=False)
            start_index=np.where(t_filtered==0.4)[0][0]
        v_filtered=v_filtered[start_index:]
        t_filtered=t_filtered[start_index:]
        i_filtered=i_filtered[start_index:]
        #save the firsdt bin of the first sweep as a reference RMP to normalise to
        #represent each trace as bins of 10 seconds - take np.mean of each 10 seconds
        if max(t_filtered) >10:
            num_bins = int(max(t_filtered)/10)  # 10 seconds worth of samples
            bin_size = len(v_filtered) // num_bins
        else: 
            num_bins=1
            bin_size = len(v_filtered)
        
        for bin_num in range(num_bins):
            # Extract the bin data
            bin_start = bin_num * bin_size
            bin_end = bin_start + bin_size
            bin_v = v_filtered[bin_start:bin_end]
            
            # Compute RMP for this bin
            rmp_mean = np.mean(bin_v)
            
            # Compute time_unix for this bin
            time_unix = row['time_unix'] + (bin_num * 10)
            
            #write data into the output:
            output_row = {'cell': row['cell'], 'group': row['group'], 'exp_type': row['exp_type'],
                    'QC': row['QC'], 'QC_extended': row['QC_extended'],'sweep_number': swp,'protocol':row['stimulus_code'],  'condition': row['condition'], 
                    'time_unix': time_unix, 'time_bin': bin_num, 'rmp_mean': rmp_mean}
                
            output = output.append(output_row, ignore_index=True)
    ref_rmp=output[~output['rmp_mean'].isna()]['rmp_mean'].iloc[0]
    output['rmp_norm']=output['rmp_mean']/ref_rmp
    output['time_diff_s']=round(output['time_unix']-output['time_unix'].iloc[0])
    output['time_diff_min']=round(output['time_diff_s']/60, 1)
    fig, axes = plt.subplots(1, 2, figsize=(15,10))
    fig.suptitle(f"{cell} - RMP change", fontsize=16, y=0.98)
    ax=axes[0]
    sns.lineplot(data=output, x='time_diff_min',marker='o', y='rmp_mean',hue='condition',palette=condition_palette, linestyle='--', ax=ax)
    ax.set_xlabel('Time (min)')
    ax.set_title('absolute')
    ax1=axes[1]
    sns.lineplot(data=output, x='time_diff_min',marker='o', y='rmp_norm',hue='condition',palette=condition_palette, linestyle='--', ax=ax1)
    ax1.set_xlabel('Time (min)')
    ax1.set_title('relative')
    plt.savefig(f'{ifolder}/{cell}_rmp_change.eps')        
    return output



#%% analyse the data - load in the data
cell=all_cells[293] #define the cell
print(f'{cell}')
cell_metadata=overview[overview.cell == cell]
s=find_correct_file(cell, all_files)
dataset, rectype = load_data(path, s, all_files)

ifolder=f'{savepath}/{cell}' #create a directory for the folder to contain cell_level data
if not os.path.exists(ifolder):
    os.makedirs(ifolder)
else:
    print(f"The directory {ifolder} already exists.")

#%% load in and process the sweep table
if rectype == 'mono':
    if os.path.exists(f'{ifolder}/{cell}_sweeps.csv'):
        sweeps=pd.read_csv(f'{ifolder}/{cell}_sweeps.csv')
    else:
        sweeps = process_mono_sweep_table(dataset,s,cell,overview,ref=0)
else:
    if os.path.exists(f'{ifolder}/{cell}_sweeps.csv'):
        sweeps=pd.read_csv(f'{ifolder}/{cell}_sweeps.csv')
    else:
        all_sweeps_from_file = process_multi_sweep_table(dataset,rectype,s,cell,overview,ref=0)
        sweeps=all_sweeps_from_file[cell]
        for sub_cell, sub_sweeps in all_sweeps_from_file.items():
            sub_folder=f'{savepath}/{sub_cell}' #create a directory for the folder to contain cell_level data
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            else:
                print(f"The directory {sub_folder} already exists.")
    
            sub_sweeps_path = f'{sub_folder}/{sub_cell}_sweeps.csv'
            if not os.path.exists(sub_sweeps_path):  # Avoid overwriting existing data
                sub_sweeps.to_csv(sub_sweeps_path)
            
#%%assess the data
for protocol in tqdm(sweeps.stimulus_code.unique(),desc='Plotting protocols', unit='protocol'):
    print(f'{protocol}')
    protocol_table=sweeps[sweeps.stimulus_code == protocol]
    mode=protocol_table.clamp_mode.iloc[0]
    frst_swp=protocol_table.sweep_number.iloc[0]
    last_sweep=protocol_table.sweep_number.iloc[-1]
    plt.figure()
    plt.title(f'{cell} {protocol}')
    for swp in tqdm(protocol_table.sweep_number,desc='Plotting sweeps', unit='sweep'):
        if rectype == 'mono':
            if mode == 'CurrentClamp':
                plt.plot(dataset.sweep(swp).v)
                plt.savefig(f'{ifolder}/{cell}_{protocol}.eps')
            else:
                plt.plot(dataset.sweep(swp).i)
                plt.savefig(f'{ifolder}/{cell}_{protocol}.eps')
        else:
            key_name=protocol_table[protocol_table.sweep_number==swp]['key_name'].iloc[0]
            swptrace= dataset['acquisition'][key_name]['data']
            plt.plot(swptrace)
            plt.savefig(f'{ifolder}/{cell}_{protocol}.eps')

del protocol, protocol_table, mode, frst_swp, last_sweep, swp
#%%analyse the data fr database - spikes data
sweeps, cell_spikes_database = analyse_CC_spikes(sweeps,dataset,cell,overview,rectype,for_database=True)
cell_spikes_database=cell_spikes_database[~cell_spikes_database.sweep_number.isna()]
sweeps['QC_spikes'] = sweeps['sweep_number'].map(cell_spikes_database.set_index('sweep_number')['QC_spikes'].to_dict())
sweeps, cell_sags_database =analyse_CC_sag(sweeps,dataset,cell,overview, rectype,for_database=True)  
sweeps, cell_spikes_database, cell_sags_database = assign_sets_after_analysis(sweeps, cell_spikes_database, cell_sags_database)
ratios_rin_database=get_fitted_features(cell_sags_database[cell_sags_database.QC_extended=='included'],sweeps,ifolder, condition_palette=condition_palette, plot_figure=True)
ap_features_database=create_excitability_summary_table(cell_spikes_database)
#%% save database data
cell_spikes_database.to_csv(f'{ifolder}/{cell}_spikes_database.csv')
cell_sags_database.to_csv(f'{ifolder}/{cell}_sags_database.csv')
ratios_rin_database.to_csv(f'{ifolder}/{cell}_fitted_features_database.csv')
ap_features_database.to_csv(f'{ifolder}/{cell}_ap_features_database.csv')
#%%analyse the data - spikes data
sweeps, cell_spikes = analyse_CC_spikes(sweeps,dataset,cell,overview,rectype)
cell_spikes=cell_spikes[~cell_spikes.sweep_number.isna()]
sweeps['QC_spikes'] = sweeps['sweep_number'].map(cell_spikes.set_index('sweep_number')['QC_spikes'].to_dict())
# for swpnr in cell_spikes.sweep_number: 
#     qc=cell_spikes[cell_spikes.sweep_number == swpnr]['QC_spikes'].iloc[0]
#     sweeps.at[sweeps.sweep_number ==swpnr, 'QC_spikes'] =qc
#%% plot the spikes
ispikes=f'{ifolder}/spikes' #create a directory for the folder to contain cell_level data
if not os.path.exists(ispikes):
    os.makedirs(ispikes)
else:
    print(f"The directory {ifolder} already exists.")


for sweepnr in tqdm(cell_spikes.sweep_number.unique(),desc='Plotting spikes', unit='sweep'):
    if rectype == 'mono':
        sweep=dataset.sweep(sweepnr)
        sweepv=sweep.v
        fs=sweep.sampling_rate
        filt_v = filter_the_signal(sweepv, fs)
    else:
        row=sweeps[sweeps.sweep_number == sweepnr]
        key_name=row['key_name'].iloc[0]        
        indexx=key_name.split('_')[1]
        hs=key_name[-1]
        sweepv=dataset['acquisition'][key_name]['data']
        sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
        fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
        filt_v = filter_the_signal(sweepv, fs)
        
    condition=cell_spikes[cell_spikes.sweep_number==sweepnr]['condition'].iloc[0]
    cmode=cell_spikes[cell_spikes.sweep_number==sweepnr]['clamp_mode'].iloc[0]
    plt.figure()
    plt.title(f'{cell} sweep {sweepnr} {condition} {cmode}')
    plt.plot(filt_v, color=condition_palette[condition])
    sweep_spikes=cell_spikes[cell_spikes.sweep_number==sweepnr]
    if not sweep_spikes.empty:
        # Convert threshold and peak indices to lists of integers
        th_ids = sweep_spikes.threshold_index.dropna().astype(int).tolist()
        peak_ids = sweep_spikes.peak_index.dropna().astype(int).tolist()
        # Plot each threshold point
        for th_idx in th_ids:
            if 0 <= th_idx < len(filt_v):  # Ensure index is valid
                plt.plot(th_idx, filt_v[th_idx], 'x', color='red')
        # Plot each peak point
        for peak_idx in peak_ids:
            if 0 <= peak_idx < len(filt_v):  # Ensure index is valid
                plt.plot(peak_idx, filt_v[peak_idx], 'x', color='black')
    plt.show()
    plt.savefig(f'{ispikes}/{cell}_sweep{sweepnr}.eps')
    plt.close()

del fs, filt_v, condition, cmode, sweep_spikes, th_ids, peak_ids, th_idx, peak_idx
#%%analyse the data - sags data
sweeps, cell_sags =analyse_CC_sag(sweeps,dataset,cell,overview, rectype)  
# plot the sags
isags=f'{ifolder}/sags' #create a directory for the folder to contain cell_level data
if not os.path.exists(isags):
    os.makedirs(isags)
else:
    print(f"The directory {ifolder} already exists.")

for sweepnr in tqdm(cell_sags.sweep_number.unique(),desc='Plotting sags', unit='sweep'):
    if rectype == 'mono':
        sweep=dataset.sweep(sweepnr)
        fs=sweep.sampling_rate
        filt_v = filter_the_signal(sweep.v, fs)
    else:
        row=sweeps[sweeps.sweep_number == sweepnr]
        key_name=row['key_name'].iloc[0]        
        indexx=key_name.split('_')[1]
        hs=key_name[-1]
        sweepv=dataset['acquisition'][key_name]['data']
        sweepi=dataset['stimulus']['presentation'][f'data_{indexx}_DA{hs}']['data']
        fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
        filt_v = filter_the_signal(sweepv, fs)
        
    condition=cell_sags[cell_sags.sweep_number==sweepnr]['condition'].iloc[0]
    cmode=cell_sags[cell_sags.sweep_number==sweepnr]['clamp_mode'].iloc[0]
    plt.figure()
    plt.title(f'{cell} sweep {sweepnr} {condition} {cmode}')
    plt.plot(filt_v, color=condition_palette[condition])
    sweep_sags=cell_sags[cell_sags.sweep_number==sweepnr]
    if not cell_sags.empty:
        plt.axhline(sweep_sags.baseline_v.iloc[0], color='blue', linestyle='--')
        plt.axhline(sweep_sags.sag_volt.iloc[0], color='orange', linestyle='--')
        plt.axhline(sweep_sags.ss_volt.iloc[0], color='grey', linestyle='--')
    plt.savefig(f'{isags}/{cell}_sweep{sweepnr}.eps')
    plt.close()

del sweepnr, fs, filt_v, condition, cmode, sweep_sags
#%% assign the sets if there are multiple Steps recorded, to sweeps, sags and spikes
sweeps, cell_spikes, cell_sags = assign_sets_after_analysis(sweeps, cell_spikes, cell_sags)
ratios_rin=get_fitted_features(cell_sags[cell_sags.QC_extended=='included'],sweeps,ifolder, condition_palette=condition_palette, plot_figure=True)
ap_features=create_excitability_summary_table(cell_spikes[cell_spikes.QC_extended=='included'])
# rmp_change=calculate_rmp_change(sweeps)
try:
    rmp_final=cont_rmp(sweeps,dataset)
    rmp_final.to_csv(f'{ifolder}/{cell}_rmp_final.csv', index=False)
except Exception as e:
    print(f'{e}')
    
    
#chirp, if present
if any('chirp' in str(code).lower() for code in sweeps.stimulus_code.unique()):
    fres, _=res_freq(dataset, sweeps, rectype, cell)
    fres.to_csv(f'{ifolder}/{cell}_fres.csv', index=False)
    if sweeps.group.iloc[0] != 'Mouse':
        fres_database = fres[fres.condition == 'baseline']
        fres_database.to_csv(f'{ifolder}/{cell}_fres_database.csv', index=False)

# if 'rmp' in overview[overview.cell==cell]['experiment'].iloc[0]:
#     rmp_measure=cont_rmp(sweeps,dataset)
#     rmp_measure.to_csv(f'{ifolder}/{cell}_rmp_binned.csv')
    
if 'stim_xx' in overview[overview.cell==cell]['experiment'].iloc[0]:
    fres_set, _= res_freq_sets(dataset, sweeps, rectype, cell)
    fres_set.to_csv(f'{ifolder}/{cell}_fres_set.csv', index=False)
    

#%% save the data
sweeps.to_csv(f'{ifolder}/{cell}_sweeps.csv', index=False)
cell_spikes.to_csv(f'{ifolder}/{cell}_spikes.csv', index=False)
cell_sags.to_csv(f'{ifolder}/{cell}_sags.csv', index=False)
ratios_rin.to_csv(f'{ifolder}/{cell}_fitted_features.csv', index=False)
ap_features.to_csv(f'{ifolder}/{cell}_ap_features.csv', index=False)



#%%
cell='M24.29.05472.61.03'
ifolder=f'{savepath}/{cell}'
cell_metadata=overview[overview.cell == cell]
sweeps=pd.read_csv(f'{ifolder}/{cell}_sweeps.csv', index_col=0)
cell_spikes=pd.read_csv(f'{ifolder}/{cell}_spikes.csv', index_col=0)
cell_sags=pd.read_csv(f'{ifolder}/{cell}_sags.csv', index_col=0)
ratios_rin=pd.read_csv(f'{ifolder}/{cell}_fitted_features.csv', index_col=0)
ap_features=pd.read_csv(f'{ifolder}/{cell}_ap_features.csv', index_col=0)

