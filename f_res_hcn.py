#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:10:28 2025

@author: annagalakhova
"""
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
from statsmodels.stats.multitest import multipletests
import logging
from tqdm import tqdm
import pickle  
current_date=datetime.now().date()
%matplotlib qt
#%%functions 
#analyse and look at the resonance and the sag vs voltage fitted slope (ref ACh-HCn resonanace in primate hippocampus paper)
#identify in whic cells I have there protocols
def find_file(root_folder, filename):
    for dirpath, _, filenames in os.walk(root_folder):
        if filename in filenames:
            return os.path.join(dirpath, filename)  # Returns full path if found
    return None

def convert_timestamp_to_human(timestamp):
    igor_to_unix_offset = 2082844800
    unix_timestamp=timestamp-igor_to_unix_offset
    output=datetime.fromtimestamp(unix_timestamp)
    return output

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
       if 'bias_current' in dataset['acquisition'][key_name].keys():
           holding = dataset['acquisition'][key_name]['bias_current'][0]
       elif 'bridge_balance' in dataset['acquisition'][key_name].keys():
           bbalance = dataset['acquisition'][key_name]['bridge_balance'][0]    
       else:
           holding = np.nan
           bbalance=np.nan
       sweeps.at[index, 'leak_pa']= holding 
       sweeps.at[index, 'bridge_balance']= bbalance 
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
            
       v=sweepv[: tp_end]
       i=sweepi[: tp_end]
       
       try:
            bsl = np.mean(i[tp_start-int(fs*0.002):tp_start-int(fs*0.001)])
            peak, peak_id = max(i, key=abs), np.argmax(np.abs(i))
            tp_inp= round(max(v, key=abs))
            if row.clamp_mode == "VoltageClamp":
                ra=tp_inp/(peak-bsl)*1000
                sweeps.at[index, 'Ra'] = ra
                #calculate tau
                y=i[peak_id:]
                x = np.arange(0, len(y))
                m_guess = 500
                b_guess = -100
                t_guess = 0.03
        
                if len(y) !=0:
                    try:
                        (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess])
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
           all_sweeps[key] = group
        
       
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
            #ASSIGN RA_NORM
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
            sets=table['stimulus_code'].str.contains('A1_Test', case=False)
            #how many sets there are? 
            sets_amount = sets.sum()
            #indexes of the sets starts
            location = sets[sets == True].index
            #locate the set_number at the index in the original df
            table.loc[location, 'set_number'] = range(1, sets_amount + 1)
            table['set_number'].fillna(method='ffill', inplace=True)
            
            for condition in unique_conditions: 
                start_condition = table_metadata[table_metadata['condition'] == condition]['fswp_condition'].iloc[0]
                table.at[table['sweep_number'] >= start_condition,'condition'] = condition

            qckech = np.nan
            table['QC'] = qckech
            table.loc[(table['Ra_norm'] >= 0.7) & (table['Ra_norm'] <= 1.3), 'QC'] = 'included'
            table.loc[(table['Ra_norm'] < 0.7) | (table['Ra_norm'] > 1.3), 'QC'] = 'excluded'
            table['QC'].fillna(method='ffill', inplace=True)
            table['QC'].fillna(method='bfill', inplace=True)
            all_sweeps[key]=table
    
    return all_sweeps    

def monoExp(x, m, k, b):
    return m * np.exp(-k * x) + b

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


def n_cell_in_file(s):
    cells = re.findall(r'0\d', s.split('.')[-2])
    slice_name= s.split('.')[:-2]
    full_names=['.'.join(slice_name+[i]) for i in cells]
    return len(cells), full_names

def res_freq(dataset, sweeps, rectype, cell):
    #get the data from the dataset file
    plt.figure()
    if rectype == 'mono':
        df = sweeps[(sweeps.stimulus_code.str.contains('CHIRP'))&(sweeps.condition.isin(['baseline', 'naag']))].reset_index(drop=True)
    else:
        if 'key_name' not in sweeps.columns or sweeps.cell.iloc[0] != cell:
            all_sweeps_from_file = process_multi_sweep_table(dataset,rectype,s,cell,overview,ref=0)
            sweeps=all_sweeps_from_file[cell]
        df = sweeps[(sweeps.stimulus_code.str.contains('CHIRP'))&(sweeps.condition.isin(['baseline', 'naag']))].reset_index(drop=True)
        
        
    output=pd.DataFrame(columns = ['cell','condition','max_impedance', 'f_res', '3db_cutoff'])
    
    data = {cell: {'baseline': {'avg_trace': None,'impedance': None,'norm_impedance': None},
            'naag': {'avg_trace': None,'impedance': None,'norm_impedance': None}}}

    
    if df.shape[0] != 0 and len(df.condition.unique())> 1:
        baseline_chirps=[]
        naag_chirps=[]
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
                
            condition=row.condition
            #get all the sweeps per condition to get the average
            if fs != 25000: #make sure they are all samples at 25000 (ipfx does 2000, I do 25000 + filtering at lowpass40: https://github.com/AllenInstitute/ipfx/blob/master/ipfx/chirp.py)
                v=downsample(dataset,row.sweep_number,rectype, sweeps,down_rate=25000)
                i=downsample(dataset,row.sweep_number,rectype, sweeps,down_rate=25000, mode='i')
                t=downsample(dataset,row.sweep_number,rectype, sweeps,down_rate=25000, mode='t')
            #make sure the sweep contains entire stimulation
            if max(t) !=  23.59996 and max(t)>20.6:
                if min_length is None or len(v) < min_length:
                    min_length=len(v)
            elif max(t) <  20.6: #as in ipfx (time until the recrodinggoes)
                continue
            
            if max(v)>-45: 
                continue
            
            #filter the trace
            v=filter_the_signal(v, fs,40,3)
            
            if condition == 'baseline':
                baseline_chirps.append(v)
            elif condition == 'naag':
                naag_chirps.append(v)

            stim_traces.append(i)
            time_traces.append(t)
 

        #process the averaged data
        if min_length is not None:
            adjusted_baseline_chirps= [arr[:min_length] for arr in baseline_chirps]
            avg_baseline = np.mean(adjusted_baseline_chirps, axis=0)
            
            adjusted_naag_chirps= [arr[:min_length] for arr in naag_chirps]
            avg_naag=np.mean(adjusted_naag_chirps, axis=0)
            
            stim_traces = [arr[:min_length] for arr in stim_traces]
            avg_stim = np.mean(stim_traces, axis=0)
            
            time_traces = [arr[:min_length] for arr in time_traces]
            avg_time = np.mean(time_traces, axis=0)
        else:
            avg_baseline = np.mean(baseline_chirps, axis=0)
            avg_naag=np.mean(naag_chirps, axis=0)
            avg_stim = np.mean(stim_traces, axis=0)     
            avg_time=np.mean(time_traces, axis=0)     
        
        stim = avg_stim
        t=avg_time
        data[cell]['baseline']['avg_trace'] = avg_baseline
        data[cell]['naag']['avg_trace'] = avg_naag
        traces={'baseline': avg_baseline, 'naag':avg_naag}
        
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
            #calculate empedance, 3db_cutoff, f_resonance
            max_imp=max(fzoomed_y)
            normalized = fzoomed_y/max_imp
            f_res = zoomed_f[np.argmax(fzoomed_y)]
            cutoff_3db=zoomed_f[np.where(normalized >= np.sqrt(0.5))[0][-1]]
            #plot the outcome
            plt.plot(zoomed_f, normalized)
            plt.axhline(y=np.sqrt(0.5), linewidth=2, linestyle=':', color='k')
            plt.axvline(x=f_res, color='red', linestyle='--')
            plt.ylabel('Impedance (normalized)', fontsize=12)
            plt.xlabel('Frequency (Hz)', fontsize=12)
            plt.title(cell)
            output=output.append({'cell':cell,'condition':k, 'max_impedance': max_imp,'f_res':f_res, '3db_cutoff': cutoff_3db}, ignore_index=True)
            data[cell][k]['impedance'] = fzoomed_y
            data[cell][k]['zoomed_f'] = zoomed_f
            data[cell][k]['norm_impedance'] = normalized
            
    return output, data


#%% loop over f_res
path = '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project'
chirp_output_path='/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Ih_current/chirps'
all_cells_df = pd.DataFrame(columns=['cell', 'condition', 'max_impedance', 'f_res', '3db_cutoff'])
all_files = [f for f in os.listdir(os.path.join(path, 'all_data')) if f.endswith('.nwb')]
all_cells_data = {} 
errors={}

overview_path = os.path.join(path, 'all_data', 'all_data_overview.xlsx')
overview = pd.read_excel(overview_path, sheet_name=-1)
cells=overview.cell.unique()


chirp_cells=[]

errors={}
for cell in tqdm(cells, desc='Processing cells', unit='cell'): 
    print(f'-------checking cell {cell}---------')
    swps_s = f'{cell}_sweeps.csv'
    swps_path = find_file(path, swps_s)
    
    if swps_path:  # Ensure the file was found before trying to read it
        sweeps = pd.read_csv(swps_path)
        chirp_protocols = sweeps[sweeps.stimulus_code.str.contains('CHIRP')]
        # Check for CHIRP protocols
        try:
            conds=len(chirp_protocols.condition.unique())
        except Exception as e: 
            print(f'{e}')
            continue
        if conds > 1:
            if cell not in all_cells_data.keys():
                #chirp_cells.append(cell)
                print(f'analysing CHIRP for cell {cell}')
                s = find_correct_file(cell, all_files)    
                dataset,rectype = load_data(path, s, all_files)
                print(f'data loaded cell {cell}')
                try:
                # Get results from your function
                    resfreq_data, cell_data = res_freq(dataset, sweeps, rectype, cell)
                    print(f'data analysed cell {cell}')
                    # Append DataFrame results
                    all_cells_df = pd.concat([all_cells_df, resfreq_data], ignore_index=True)
                    print(f'data stored cell {cell}')
                    # Update master dictionary with this cell's data
                    all_cells_data.update(cell_data)
                    print(f'traces loaded cell {cell}')
                    plt.savefig(f'{chirp_output_path}/{cell}_impedance.eps', format='eps')
                except Exception as e:
                    print(f'defect in cell {cell}, moving on to next cell')
                    errors[cell]=e
            else:
                continue
        else:
            print(f'no CHIRP for cell {cell}')
            continue


#%%
# Example dictionary and data

all_cells_df.to_csv(f'{chirp_output_path}/all_cells_res_{current_date}.csv')
data_to_save = {"all_cells_data": all_cells_data, "all_cells_df": all_cells_df}
with open(f"{chirp_output_path}/data_dict_{current_date}.pkl", "wb") as f:
    pickle.dump(data_to_save, f)
    
# Open the file in read mode ("rb")
with open(f"{chirp_output_path}/data_dict_{current_date}.pkl", "rb") as f:
    data_to_load = pickle.load(f)



#%% plot the dat per cell in a figure
condition_palette={'baseline': '#009FE3','naag': '#F3911A', 'vehicle': '#706F6F'}

for cell in all_cells_data.keys():
    
    cell_data = all_cells_data[cell]
    cell_results = all_cells_df[all_cells_df.cell == cell]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig.suptitle(f'CHIRP response in cell {cell}')  # Corrected title placement
    
    baseline_start = cell_data['baseline']['avg_trace'][0]
    
    for condition in cell_data.keys():
        f_res = cell_results[cell_results.condition == condition]['f_res'].iloc[0]
        avg_trace = cell_data[condition]['avg_trace']
        impedance = cell_data[condition]['impedance']
        norm_impedance = cell_data[condition]['norm_impedance']
        zoomed_f = cell_data[condition]['zoomed_f']

        # Plot avg_signal
        ax = axes[0]
        shifted_naag = cell_data['naag']['avg_trace'] - (cell_data['naag']['avg_trace'][0] - baseline_start)
        if condition == 'naag':
            ax.plot(shifted_naag, color=condition_palette[condition])
        else:
            ax.plot(avg_trace, color=condition_palette[condition])
            
        
        ax.set_title("Average traces")
        miny = min(np.min(cell_data['baseline']['avg_trace']), np.min(shifted_naag))
        maxy = max(np.max(cell_data['baseline']['avg_trace']), np.max(shifted_naag))
        ax.set_ylim(miny, maxy)
        

        # Plot absolute impedance
        ax = axes[1]
        ax.plot(zoomed_f, impedance, color=condition_palette[condition])
        ax.set_title("Impedance")
        miny = min(np.min(cell_data['baseline']['impedance']), np.min(cell_data['naag']['impedance']))
        maxy = max(np.max(cell_data['baseline']['impedance']), np.max(cell_data['naag']['impedance']))
        ax.set_ylim(miny, maxy)

        # Plot norm_impedance
        ax = axes[2]
        ax.plot(zoomed_f, norm_impedance, color=condition_palette[condition])
        ax.set_title("Normalised Impedance")
        ax.axvline(x=f_res, color=condition_palette[condition], linestyle='--', alpha=0.8, label=f"{condition} f_res")
        ax.set_ylim(0, 1.1)

    plt.savefig(f'{chirp_output_path}/{cell}_all_plots.eps', format='eps')

#%% plot all the cells
condition_palette={'baseline': '#009FE3','naag': '#F3911A', 'vehicle': '#706F6F'}
all_cells_df= pd.read_csv(f'{chirp_output_path}/all_cells_res_{current_date}.csv')

hbaseline_traces = []
hnaag_traces = []
mbaseline_traces = []
mnaag_traces = []
zoomed_f=all_cells_data['H23.29.244.11.63.01']['baseline']['zoomed_f']
# Extract data from all cells
for cell in all_cells_data.keys():
    dict_cell=all_cells_data[cell]   
    baseline_imp=dict_cell['baseline']['norm_impedance']
    naag_imp=dict_cell['naag']['norm_impedance']
    if baseline_imp is not None and naag_imp is not None:
        if 'M' in cell:
            mbaseline_traces.append(baseline_imp)
            mnaag_traces.append(naag_imp)
        elif 'H' in cell:
            hbaseline_traces.append(baseline_imp)
            hnaag_traces.append(naag_imp)
    
# Compute mean and standard deviation
#human traces
hbaseline_mean = np.mean(hbaseline_traces, axis=0)
hbaseline_sem = np.std(hbaseline_traces, axis=0) / np.sqrt(len(hbaseline_traces))
hnaag_mean = np.mean(hnaag_traces, axis=0)
hnaag_sem = np.std(hnaag_traces, axis=0) / np.sqrt(len(hnaag_traces))
#mouse traces
mbaseline_mean = np.mean(mbaseline_traces, axis=0)
mbaseline_sem = np.std(mbaseline_traces, axis=0) / np.sqrt(len(mbaseline_traces))
mnaag_mean = np.mean(mnaag_traces, axis=0)
mnaag_sem = np.std(mnaag_traces, axis=0) / np.sqrt(len(mnaag_traces))


#plot into one figure
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
ax = axes[0,0]
human=all_cells_df[all_cells_df['cell'].str.startswith('H')]
sns.boxplot(data=human,x='condition', y='f_res', hue='condition', palette=condition_palette, ax=ax)
ax.set_title(f'Human, N={len(human.cell.unique())}')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.8)
ax.set_ylim(0, 4)

ax = axes[0,1]
mouse=all_cells_df[all_cells_df['cell'].str.startswith('M')]
sns.boxplot(data=mouse,x='condition', y='f_res', hue='condition', palette=condition_palette, ax=ax)
ax.set_title(f'Mouse, N={len(mouse.cell.unique())}')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.8)
ax.set_ylim(0, 2)

ax = axes[1,0]
ax.plot(zoomed_f, hbaseline_mean, label="Baseline", color=condition_palette['baseline'])
ax.plot(zoomed_f, hnaag_mean, label="NAAG", color=condition_palette['naag'])
# Plot shaded region for std
ax.fill_between(zoomed_f, hbaseline_mean - hbaseline_sem, hbaseline_mean + hbaseline_sem, color=condition_palette['baseline'], alpha=0.2)
ax.fill_between(zoomed_f, hnaag_mean - hbaseline_sem, hnaag_mean + hbaseline_sem, color=condition_palette['naag'], alpha=0.2)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Frequency (Hz)")  
ax.set_ylabel("Normalised Impedance") 

ax = axes[1,1]
ax.plot(zoomed_f, mbaseline_mean, label="Baseline", color=condition_palette['baseline'])
ax.plot(zoomed_f, mnaag_mean, label="NAAG", color=condition_palette['naag'])
# Plot shaded region for std
ax.fill_between(zoomed_f, mbaseline_mean - mbaseline_sem, mbaseline_mean + mbaseline_sem, color=condition_palette['baseline'], alpha=0.2)
ax.fill_between(zoomed_f, mnaag_mean - mnaag_sem, mnaag_mean + mnaag_sem, color=condition_palette['naag'], alpha=0.2)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Frequency (Hz)")  
ax.set_ylabel("Normalised Impedance") 


plt.savefig(f'{chirp_output_path}/RESONANCE_PLOT.eps', format='eps')

#%%analyse sags

def sag(sweeps,dataset,cell,overview, rectype):
    #define the ouput table
    sag_result=pd.DataFrame(columns = ['cell','sweep_number','condition', 'clamp_mode','curr_inj','baseline_v','sag_volt',
                                       'ss_volt', 'sag_deflection','ss_deflection','tau','R2',
                                       'sag_ratio'])
    #select the correct protocol - CCsteps with clamping - analyse al the protocols there are
    df2 = sweeps[sweeps["stimulus_code"].str.contains("steps|Thres", regex=True) &
                 (sweeps["clamp_mode"] == "CurrentClamp") & sweeps["condition"].isin(["baseline", "naag"])]

    #loop over sweeps to load them in and analyse
    if df2.shape[0] != 0 and len(df2.condition.unique())>1:
        for index, row in df2.iterrows():
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
                    condition=row.condition
                    curr_inj = round(max(sweepi[stim_start: stim_end],key=abs))
                    #filter
                    #check if the trace has nan values and take them out (most likely for the PS protools which trim randomly)
                    nan_indices = np.where(np.isnan(sweepv))[0]
                    if len(nan_indices) > 0:  # If there are NaN values
                        first_nan_idx = nan_indices[0]
                        trimmed_sweepv = sweepv[:first_nan_idx]
                        filt_v = filter_the_signal(trimmed_sweepv, fs, cutoff=500)
                    else:  # If no NaN values
                        filt_v = filter_the_signal(sweepv, fs, cutoff=500)
                        
                    #if sweep has negative input, analyse sag and calculate Rin - fit in the curve into input-output
                    if curr_inj < 0:
                       sag_ind = np.argmin(filt_v)
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
                                (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess])
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
                    
                       sag_result= sag_result.append({'cell':row.cell, 'sweep_number': row.sweep_number,'condition':condition,
                                                      'clamp_mode':cmode,'curr_inj':curr_inj, 'baseline_v':baseline,'sag_volt':sagv,
                                                      'ss_volt':ss_volt, 'sag_deflection':sag_deflection, 'ss_deflection':ss_deflection,
                                                       'tau':tau, 'R2': r2,'sag_ratio':sag_ratio}, ignore_index=True)
                    elif curr_inj == 0 or curr_inj >0: 
                        continue
                except Exception as e:
                    print(f'sweep {row.sweep_number} is not good : {e}')
                    continue  # Skip this sweep if error occur
    return sag_result

def get_fitted_features(df):
    output=pd.DataFrame(columns=['cell', 'clamping', 'condition','Rin', 'ss_slope', 'sag_slope', 'ratio_of_slopes'])
    
    if len(df.condition.unique()) != 2:
       print(f"Cell {df.cell.iloc[0]} missing condition pair")
       return output
    
    clamping = df.clamp_mode.iloc[0]
    cell=df.cell.iloc[0]
    
    for condition in df.condition.unique():
        df_cond=df[df.condition == condition]
        coeffs_rin = np.polyfit(list(df_cond['curr_inj']),list(df_cond['sag_deflection']), 1)
        rin_fitted = abs(coeffs_rin[0])*1000
        ss_slope=np.polyfit(list(df_cond['curr_inj']),list(df_cond['ss_volt']), 1)[0]
        sag_slope=np.polyfit(list(df_cond['curr_inj']),list(df_cond['sag_volt']), 1)[0]
        ratio_of_slopes=sag_slope/ss_slope
        
        new_row = pd.DataFrame({'cell': [cell],'clamping': [clamping],'condition': [condition],
           'Rin': [rin_fitted],'ss_slope': [ss_slope],'sag_slope': [sag_slope],'ratio_of_slopes': [sag_slope/ss_slope]})
        output = pd.concat([output, new_row])
        
        
    return output

#%%analyse sags, add rin and sag_ratio_slopes (posthoc)
output_path='/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Ih_current/sags'
for cell in tqdm(overview.cell.unique(), desc='Processing cells', unit='cell'): 
    
    if os.path.exists(f'{output_path}/{cell}_sags.csv'):
        print(f'Skipping {cell} - CSV already exists')
        continue
    
    swps_s = f'{cell}_sweeps.csv'
    swps_path = find_file(path, swps_s)
    if swps_path:  # Ensure the file was found before trying to read it
        sweeps = pd.read_csv(swps_path)
    else:
        print(f'no sweeps found for cell {cell}')
        continue
        
    #check if there is such data at all - we need any kind of dataa, comparing results we will do later
    subset_sweeps = sweeps[
                            (sweeps['stimulus_code'].str.contains('steps|Thre', regex=True))]
    if subset_sweeps.shape[0] == 0:
        print(f'!!!!!!!!!!!!!!!!!!!      no data for cell {cell}, continue !!!!!!!!!!!!!!')
        continue
    if condition not in subset_sweeps.columns:
        print(f'!!!!!!!!!!!!!!!!!!!      no data for cell {cell}, continue !!!!!!!!!!!!!!')
        continue
    
    elif len(subset_sweeps.condition.unique()) >1:
        print(f'analysing cell {cell}!!!!')     
        dataset, rectype = load_data(path, find_correct_file(cell, all_files), all_files)
        print(f'LOADED DATA FOR cell {cell}!!!!') 
        sags=sag(sweeps,dataset,cell,overview, rectype)
        print(f'ANALYSED DATA FOR cell {cell}!!!!')
        sags.to_csv(f'{output_path}/{cell}_sags.csv')



#%% compare the results of the fitting clamped vs not clamped: Rin, ss_slope and sag_slope
all_files = [f for f in os.listdir(output_path) if f.endswith('_sags.csv')]
comparison=pd.DataFrame(columns=['cell', 'clamping', 'Rin', 'ss_slope', 'sag_slope'])
clamped_data=pd.DataFrame()
not_clamped_data=pd.DataFrame()
for file in all_files:
    cell=file[:-9]
    swps_s = f'{cell}_sweeps.csv'
    swps_path = find_file(path, swps_s)
    if swps_path:  # Ensure the file was found before trying to read it
           sweeps = pd.read_csv(swps_path)
    
    sag_i=pd.read_csv(f'{output_path}/{file}')
    sag_i=sag_i.merge(sweeps[['sweep_number', 'condition', 'stimulus_code']], on='sweep_number', how='left')
    
    #make sure to only include one protocol, ideally steps, else Threshold
    for protocol in ['teps', 'thre']:
        protocol_df = sag_i[sag_i['stimulus_code'].str.contains(protocol)]
        if len(protocol_df) > 0 and len(protocol_df['condition'].unique()) == 2:
            sag_i = protocol_df
            break
    else:
        print(f'No paired data for cell {cell}')
        continue
        
    for mode in ['clamped', 'not_clamped']:
           mode_data = sag_i[sag_i.clamp_mode == mode]
           if len(mode_data) > 0:
               first_instances = mode_data.groupby(['condition', 'curr_inj']).first().reset_index()
               mode_features=get_fitted_features(mode_data)
               comparison=pd.concat([comparison,mode_features])
               if mode == 'clamped':
                   clamped_data = pd.concat([clamped_data, first_instances])
               else:
                   not_clamped_data = pd.concat([not_clamped_data, first_instances])

cell_groups = overview.groupby('cell')['group'].first()
cell_qcs = overview.groupby('cell')['QC'].first()
# Add group to comparison
comparison['group'] = comparison['cell'].map(cell_groups)
comparison['QC'] = comparison['cell'].map(cell_qcs)
comparison=comparison[comparison.QC == 'included']


comparison.to_csv(f'{output_path}/ALLALL_features_{current_date}.csv')
clamped_data.to_csv(f'{output_path}/all_clamped_sags_{current_date}.csv')
not_clamped_data.to_csv(f'{output_path}/all_not_clamped_sags_{current_date}.csv')
    
cells_with_both = comparison.groupby('cell')['clamping'].nunique() == 2
cells_with_both = cells_with_both[cells_with_both].index


# Filter comparison dataframe
comparison_both_modes = comparison[comparison['cell'].isin(cells_with_both)]

comparison_both_modes.to_csv(f'{output_path}/test_clampedVSnot_{current_date}.csv')

#make a datafeamr with clamped+ not clamped data, take clamed from those who have both
cells_single = comparison[~comparison['cell'].isin(cells_with_both)]
cells_both_clamped = comparison[(comparison['cell'].isin(cells_with_both)) & 
                              (comparison['clamping'] == 'clamped')]
comparison_merged =pd.concat([cells_both_clamped, cells_single])
comparison_merged.to_csv(f'{output_path}/test_clamped+not_{current_date}.csv')

#%% select sweep which is closest to -100

#%% plot CCsteps
# Generate color map based on condition
'baseline': '#009FE3','naag': '#F3911A'

import matplotlib.colors as mcolors

#plot baseline sweeps
sweep_numbers=range(48,59)

base_color = '#009FE3'  # Your specific blue
rgb = mcolors.hex2color(base_color)
colors = [(*rgb, alpha) for alpha in np.linspace(1.0, 0.3, len(sweep_numbers))]

for i, sweep_num in enumerate(sweep_numbers):
   sweep = dataset.sweep(sweep_num)
   plt.plot(sweep.t, sweep.v, color=colors[i], zorder=i)

plt.ylim([-90,-73]) 
plt.savefig(f'{output_path}/example_baseline_humanl3.pdf', dpi=300, transparent=True,format='pdf')

#plot naag_sweeps
plt.figure()
sweep_numbers=range(211,222)

base_color='#F3911A'
rgb = mcolors.hex2color(base_color)
colors = [(*rgb, alpha) for alpha in np.linspace(1.0, 0.3, len(sweep_numbers))] 

for i, sweep_num in enumerate(sweep_numbers):
   sweep = dataset.sweep(sweep_num)
   plt.plot(sweep.t, sweep.v, color=colors[i])
plt.ylim([-90,-73]) 
plt.savefig(f'{output_path}/example_naag_humanl3.pdf', dpi=300, transparent=True,format='pdf')



#%% plot slopes for baseline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cell_groups = overview.groupby('cell')['group'].first()
cell_qcs = overview.groupby('cell')['QC'].first()
# Add group to comparison
clamped_data['group'] = clamped_data['cell'].map(cell_groups)
clamped_data['QC'] = clamped_data['cell'].map(cell_qcs)
clamped_data=clamped_data[clamped_data.QC == 'included']

# Calculate means and standard errors for each current injection value
def get_stats(df, condition, value_col):
    stats = df[df['condition'] == condition].groupby('curr_inj')[value_col].agg(['mean', 'std', 'count']).reset_index()
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    return stats['curr_inj'], stats['mean'], stats['sem']


mouse_cells=clamped_data[clamped_data.cell.str.startswith('H')]
# Create figure
fig, ax = plt.subplots(figsize=(5, 6))

# Plot for baseline condition
curr_inj, sag_mean, sag_sem = get_stats(mouse_cells, 'baseline', 'sag_deflection')
curr_inj, ss_mean, ss_sem = get_stats(mouse_cells, 'baseline', 'ss_deflection')

sag_mean = -sag_mean
ss_mean = -ss_mean

# Plot sag voltage with error bars
ax.errorbar(curr_inj, sag_mean, yerr=sag_sem,
          color='#009FE3', marker='o', markersize=6, 
          linestyle='-', linewidth=1.5,
          label='Sag voltage', capsize=3,
          markerfacecolor='#009FE3')

# Plot steady state voltage with error bars
ax.errorbar(curr_inj, ss_mean, yerr=ss_sem,
          color='#009FE3', marker='o', markersize=6,
          linestyle='-', linewidth=1.5,
          label='SS voltage', capsize=3,
          markerfacecolor='white')

# Customize plot
ax.set_xlabel('Current injection (pA)', fontsize=12)
ax.set_ylabel('Voltage (mV)', fontsize=12)
#ax.set_ylim(max(max(sag_mean), max(ss_mean))*1.1, min(min(sag_mean), min(ss_mean))*1.1)  # Set y limits
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.show()
