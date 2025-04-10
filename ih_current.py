#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:47:23 2024

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
import h5py
sys.path.append('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/scripts')
# import toolkit
# from toolkit import *
from tqdm import tqdm 


#%% functions

#exponential to fit the tau data
def monoExp(x, m, k, b):
    return m * np.exp(-k * x) + b

#low pass butter filter to filter the data
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
     # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def plot_ih_steps_and_tails(traces, traces_tail, df2, savedir):
    conditions = df2['condition'].unique()
    compensations = df2['compensation'].unique()
    cell=df2.cell[0]
    
    for condition in conditions:
        for compensation0 in compensations:
            if compensation0 == 'no':
                compensation = 'no compensation'
            else: 
                compensation = 'compensation'
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            fig.suptitle(f'{condition.capitalize()}, {compensation.capitalize()}')

            # Determine the sweeps to plot for this condition and compensation
            sweeps = df2[(df2['condition'] == condition) & (df2['compensation'] == compensation0)]['sweep']
            
            # Generate color map based on condition
            if condition == 'baseline':
                cmap = plt.get_cmap('Blues')
                color_start = 150  # Lighter blue
                color_end = 255  # Darker blue
            else:
                cmap = plt.get_cmap('Oranges')
                color_start = 150  # Lighter orange
                color_end = 255  # Darker orange

            colors = [cmap(i) for i in np.linspace(color_start, color_end, len(sweeps)).astype(int)]

            for ax, (current_traces, title) in zip(axs, [(traces, 'Ih step'), (traces_tail, 'Ih tail')]):
                for sweep, color in zip(sweeps, colors):
                    if sweep in current_traces:
                        trace = current_traces[sweep]
                        ax.plot(trace, color=color)
                ax.set_title(f'{title}')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Current (pA)')
            plt.savefig(f'{savedir}/{cell}_{condition}_{compensation}.eps', format='eps')    

    for compensation0 in compensations:
        if compensation0 == 'no':
            compensation = 'No Compensation'
        else:
            compensation = 'Compensation'

        fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        fig.suptitle(f'{compensation}')

        for condition in conditions:
            sweeps_to_plot = df2[(df2['condition'] == condition) & (df2['compensation'] == compensation0)]['sweep']

            # Determine color map and range based on condition, use a more vibrant section of the colormap
            if condition == 'baseline':
                cmap = plt.get_cmap('Blues')
                color_range = np.linspace(0.2, 0.8, len(sweeps_to_plot))  # More vibrant section of Blues
            else:
                cmap = plt.get_cmap('Oranges')
                color_range = np.linspace(0.2, 0.8, len(sweeps_to_plot))  # More vibrant section of Oranges

            colors = [cmap(i) for i in color_range]  # Adjusted for more vibrant colors

            for ax, (current_traces, title) in zip(axs, [(traces, 'Ih step'), (traces_tail, 'Ih tail')]):
                for sweep, color in zip(sweeps_to_plot, colors):
                    if sweep in current_traces:
                        trace = current_traces[sweep]
                        ax.plot(trace, color=color, label=f'{condition} sweep {sweep}')
                ax.set_title(f'{title}')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Current (pA)')

        # Add legend outside of the plot area
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        plt.savefig(f'{savedir}/{cell}_both_{compensation}.eps', format='eps')
    
    
    
    
#%%get major information
path = '/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Ih_current/row_data'
files = [f for f in os.listdir(path) if f.endswith('.nwb')]
metadata=pd.read_excel('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Experiment3_2.5_5_10_20; Gabazine, continous_recording/metadata.xlsx')
savedir = path[:-8]+'analysed_data/included/'

#%%load in all the data
data = {}
total_sweeps = {}
for name in tqdm(files, desc='Loading files', unit='file'):
    data[str(name)] = create_ephys_data_set(nwb_file=path+'/'+name) #dictionary containing all the dataset, with key = file name
    total_sweeps[str(name)] = data[str(name)].filtered_sweep_table() #dictionary containing all the raw metadta, key = file name
del name
#%%
file=files[4]
print(file)
dataset=create_ephys_data_set(nwb_file=path+'/'+file)
try:
    sweeps=dataset.filtered_sweep_table() 
    rectype='mono'
except Exception as e:
    print(f"-----------------------Error loading sweep table: {e}------------------------------")
    rectype='multi'
#%%processing the sweep table
#add conditions to the sweep table
if rectype == 'multi':
    f = h5py.File(path+'/'+file,'r')
    #get the sweeps data into the table
    sweeps=pd.DataFrame(list(f['acquisition'].keys()))
    sweeps['sweep_number'] = sweeps[0].apply(lambda x: int(x.split('_')[1]))
    sweeps['channel'] = sweeps[0].apply(lambda x: 'HS'+x.split('_')[2][-1])
    sweeps = sweeps.rename(columns={0: 'key_name'})
    #construct a sweep table
    for i in range(0,len(f['general']['labnotebook']['ITC18_Dev_0']['textualValues'])):
        temp=f['general']['labnotebook']['ITC18_Dev_0']['textualValues'][i]
        sweep_number=int(temp[0][0])
        stimulus_code=temp[5][0]
        unit=temp[7][0]
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
    if file[0]== 'H' and len(file) > 23:
        sweeps.at[sweeps['channel'] == 'HS0', 'cell'] = file[:-6]
        sweeps.at[sweeps['channel'] == 'HS1', 'cell'] = file[:-8]+file[-6:-4]
    elif file[0]== 'M' and len(file) == 22:
        sweeps.at[sweeps['channel'] == 'HS0', 'cell'] = file[:-4]
    sweeps['condition']=None
    ref_index=sweeps[sweeps['stimulus_code'] == 'spont_activity'].index[0]
    #for spont_activity
    metadata=pd.read_excel('/Users/annagalakhova/Downloads/Book2.xlsx')
    # metadata= pd.read_excel(path[:-19]+'sEPSPs/events/events_'+file[:-6]+'.xlsx')
    ref_dict={}
    for swp in metadata.sweep.unique():
        condition=metadata.loc[metadata.sweep==swp].reset_index().condition[0]
        ref_dict[swp]=condition
    for index, row in sweeps.iterrows():
        if row.sweep_number in ref_dict:
            sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'condition'] = ref_dict[row.sweep_number]
        else:
            if row.sweep_number  < ref_index:
                sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'condition'] = 'baseline'
            else:
                sweeps.at[sweeps['sweep_number'] == row.sweep_number, 'condition'] = 'naag'
#add metadata to swep table is it is a single channel recording
elif rectype == 'mono':
    sweeps['cell']= file[:-4]
    sweeps['condition']=None
    ref_index=sweeps[sweeps['stimulus_code'] == 'spont_activity'].index[0]
    #for spont_activity
    metadata= pd.read_excel(path[:-19]+'sEPSPs/events/events_'+file[:-4]+'.xlsx')
    ref_dict={}
    for swp in metadata.sweep.unique():
        condition=metadata.loc[metadata.sweep==swp].reset_index().condition[0]
        ref_dict[swp]=condition
    for index, row in sweeps.iterrows():
        if index in ref_dict:
            sweeps.at[index, 'condition'] = ref_dict[index]
        else:
            if index < ref_index:
                sweeps.at[index, 'condition'] = 'baseline'
            else:
                sweeps.at[index, 'condition'] = 'naag'    
      
#del metadata,ref_dict,condition, index, row,f,i, ref_index,swp, temp, sweep_number, stimulus_code, stimulus_unit, clamp_mode
#%% get the corresponding Ras
#add Ra and information on whole_cell_compensation to each sweep on sweep table
if rectype == 'mono':
    for index,row in sweeps.iterrows():
        if row.clamp_mode == "VoltageClamp":
            f = h5py.File(path+'/'+file,'r')
            if row.sweep_number <= 9:
                indexx=str(0)+str(index)
            else:
                indexx=row.sweep_number
            if 'whole_cell_capacitance_comp' in f['acquisition']['data_000'+str(indexx)+'_AD0'].keys(): #think of how to implement PR AD1 for the future if I figure out channels in ipfx
                sweeps.at[index, 'Ra'] = f['acquisition']['data_000'+str(indexx)+'_AD0']['whole_cell_series_resistance_comp'][0]*0.000001
                sweeps.at[index, 'compensation'] = 'yes'
            else:
                sweep = dataset.sweep(sweep_number=index)
                tp_start, tp_end = sweep.epochs['test'][0],sweep.epochs['test'][1]
                v=sweep.v[tp_start: tp_end]
                i=sweep.i[tp_start: tp_end]
                stim_index = np.nonzero(v)[0][0]
                bsl = i[stim_index-100]
                peak = max(i)
                tp_inp= round(max(v))
                ra=(tp_inp/(peak-bsl))*1000
                sweeps.at[index, 'Ra'] = ra
                sweeps.at[index, 'compensation'] = 'no'
    
    del index, row, f, indexx,  tp_start, tp_end, v, i, stim_index, bsl, peak, tp_inp, ra, sweep
elif rectype == 'multi':
    for index,row in sweeps.iterrows():
        if row.clamp_mode == "VoltageClamp":
            f = h5py.File(path+'/'+file,'r')
            print(row)
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
    del index, row, sweepi, sweepv, i, v, tp_start, tp_end, stim_index, bsl, peak, tp_inp, ra

sweeps.to_csv(savedir+file[:-4]+'_sweeps_metadata.csv')
#%% select the data to analyse
sweeps=pd.read_csv(savedir+file[:-4]+'_sweeps_metadata.csv')


if rectype == 'mono':
#sweeps = pd.read_csv(savedir+file[:-4]+'_sweeps_metadata.csv')    
    df= sweeps[sweeps.stimulus_code == 'ih'].reset_index() # define df here as a pupiece of sweeps which contains ih protocols
elif rectype == 'multi':
    df= sweeps[(sweeps.stimulus_code == 'ih') & (sweeps.channel == 'HS0')].reset_index() 


#%% analysis
signals={}
signals_tail={}
output=pd.DataFrame(columns=['cell','sweep','condition','compensation','voltage_step', 'Ih_start', 'Ih_end', 'Ih_ampl', "Ih_tau", "r2offit", 'Ih_comment',
                             'Ih_tail_start', 'Ih_tail_end', 'Ih_tail_ampl', 'Ih_tail_tau', 'r2offit_tail', 'Ih_tail_comment'])

for index, row in df.iterrows():
    try:
        if rectype == 'mono':
            #get the data for the sweep
            sweep=dataset.sweep(sweep_number=row.sweep_number)
            print (f'-----------ANALYSING SWEEP {row.sweep_number} -------------')
            fs=sweep.sampling_rate
            nyq = fs/2
            timepoints = (np.where(sweep.v[:-1] != sweep.v[1:])[0]) + 1
            
            if len (timepoints) != 6:
                temp=dataset.sweep(sweep_number=row.sweep_number-1)
                timepoints = (np.where(temp.v[:-1] != temp.v[1:])[0]) + 1 
                del temp
            i, v, t = butter_lowpass_filter(data=sweep.i, cutoff=500, fs=fs, order=3), sweep.v, sweep.t
            art = 150
        
            del sweep, nyq
            
        elif rectype == 'multi':
            sweepi = f['acquisition'][row.key_name]['data']
            sweepv=f['stimulus']['presentation'][row.key_name[:-4]+'_DA'+row.key_name[-1]]['data']
            fs=int(1/sweepi.attrs['IGORWaveScaling'][1][0]*1000)
            nyq = fs/2
            timepoints = (np.where(sweepv[:-1] != sweepv[1:])[0]) + 1
            
            if len (timepoints) != 6:
                if int(row.key_name[8:10]) <= 9:
                    temp=f['stimulus']['presentation'][row.key_name[:8]+str(0)+str(int(row.key_name[8:10])-1)+'_DA'+row.key_name[-1]]['data']
                    timepoints = (np.where(temp[:-1] != temp[1:])[0]) + 1 
                else:
                    temp=f['stimulus']['presentation'][row.key_name[:8]+str(int(row.key_name[8:10])-1)+'_DA'+row.key_name[-1]]['data']
                    timepoints = (np.where(temp[:-1] != temp[1:])[0]) + 1 
            
            i, v, t = butter_lowpass_filter(data=sweepi, cutoff=500, fs=fs, order=3), sweepv, np.arange(0,len(sweepi))/fs
            # i, v, t = moving_average(sweepi, window_size=20), sweepv, np.arange(0,len(sweepi))/fs
            art = 150

    
        voltage_step=round(np.mean(v[timepoints[3]+5:timepoints[4]]))-70
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'sweep {row.sweep_number}, voltage step {voltage_step}', fontsize=16)
    
        #perform analysis of Ih current
        ih_start,ih_end= timepoints[3], timepoints[4] #get the start and end of the epoch
        if i[ih_start+art]-i[ih_start-500]<400 and np.std(i[ih_start:ih_end])<100: #make sure your data is not spiking
            curr_start, currmaxindex =max(i[ih_start+art:ih_start+art+12500]), np.argmax(i[ih_start+art:ih_start+art+12500]) #find the peak in the first 20ms of the epoch
            signal, signalt = i[ih_start+art+currmaxindex:ih_end-art], t[ih_start+art+currmaxindex:ih_end-art]  #declare the signal peak to the end
            signals[row.sweep_number]=signal #append it to the list
            curr_ss= np.mean(signal[len(signal)-int(fs/5):]) #take ss of the current as last 200ms of the current step
            ih_ampl=curr_start-curr_ss
            
            #plot the Ih current in the first subplot with
            axs[0].plot(signalt, signal, color='k') #plot the signal
            axs[0].plot(signalt[0],signal[0],'x', color='b') #plot the curr_start
            axs[0].plot(signalt[-int(fs/5)],signal[-int(fs/5)],'x', color='c') #plot the curr_ss
            axs[0].set_title('Ih step')
            #fit in the curve
            if ih_ampl > 10:
                pars=pd.DataFrame(columns = ['k0','k', 'rSquared', 'tau'])
                for k0 in range (1,31,2):
                    p0 = (ih_ampl, k0, curr_ss) # start with values near those we expect
                    params,cv = sp.optimize.curve_fit(monoExp, signalt, signal, p0 ,maxfev=100000)
                    m,k,b = params
                    #check goodness of fit
                    squaredDiffs = np.square(signal - monoExp(signalt, m, k, b))
                    squaredDiffsFromMean = np.square(signal - np.mean(signal))
                    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                    pars= pars.append({'k0':k0,'k':k, 'rSquared': rSquared, 'tau': (1 / k)*1000},ignore_index=True)
                    if max(pars.rSquared) > 0.9:
                        result = pars[pars.rSquared == max(pars.rSquared)].reset_index()
                        p0 = (ih_ampl, result.k0[0], curr_ss) # start with values near those we expect
                        params,cv = sp.optimize.curve_fit(monoExp, signalt, signal, p0 ,maxfev=100000)
                        m,k,b = params
                        #check goodness of fit
                        squaredDiffs = np.square(signal - monoExp(signalt, m, k, b))
                        squaredDiffsFromMean = np.square(signal - np.mean(signal))
                        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                        taumsec =( 1 / k)*1000
                        axs[0].text(0.7, 0.7, f'R² = {rSquared:.2f}', ha='right', va='top',style='italic',color='k', transform=axs[0].transAxes)
                        axs[0].plot(signalt, monoExp(signalt, m, k, b), '--', color='r')
                        ih_comment = 'good fit'
                    if 0.7 < max(pars.rSquared) < 0.9:
                        result = pars[pars.rSquared == max(pars.rSquared)].reset_index()
                        p0 = (ih_ampl, result.k0[0], curr_ss) # start with values near those we expect
                        params,cv = sp.optimize.curve_fit(monoExp, signalt, signal, p0 ,maxfev=100000)
                        m,k,b = params
                        #check goodness of fit
                        squaredDiffs = np.square(signal - monoExp(signalt, m, k, b))
                        squaredDiffsFromMean = np.square(signal - np.mean(signal))
                        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                        taumsec =( 1 / k)*1000
                        axs[0].text(0.7, 0.7, f'R² = {rSquared:.2f} !!! lower than 0.95', ha='right', va='top',style='italic',color='r', transform=axs[0].transAxes)
                        axs[0].plot(signalt, monoExp(signalt, m, k, b), '--', color='r')
                        ih_comment = 'poor fit'
                    if 0.7 > max(pars.rSquared):
                        rSquared = np.nan
                        taumsec = np.nan
                        axs[0].text(0.7, 0.7, 'no fit', ha='right', va='top',style='italic',color='r', transform=axs[0].transAxes)
                        ih_comment = 'potential K current'
                        curr_start=curr_ss=ih_ampl= np.nan
            else:
                taumsec = rSquared =np.nan
                axs[0].text(0.7, 0.7, 'no current', ha='right', va='top',style='italic',color='r', transform=axs[0].transAxes)
                ih_comment = 'no current'
        else:
            axs[0].plot(t[ih_start:ih_end],i[ih_start:ih_end], color='k')#think here why and how to get rid of the huge drop before the current appears
            axs[0].set_title('Ih step')
            axs[0].text(0.7, 0.7, 'no analysis', ha='right', va='top',style='italic',color='k', transform=axs[0].transAxes)
            ih_comment = 'spikes'
            curr_start = curr_ss = ih_ampl = taumsec = rSquared =np.nan
        
        #perform analysis of Ih tail current
        tail_start,tail_end= timepoints[4], timepoints[5] #get the start and end of the epoch
        if i[tail_start+art]-i[tail_start-500]<400 and np.std(i[tail_start+art:tail_end-art])<100: #make sure your data is not spiking
            curr_starttail, currmaxindextail =max(i[tail_start+art:tail_start+art+6250]), np.argmax(i[tail_start+art:tail_start+art+6250]) #find the peak in the first 20ms of the epoch
            signal_tail, signal_tailt = i[tail_start+art+currmaxindextail:tail_end-art], t[tail_start+art+currmaxindextail:tail_end-art]  #declare the signal peak to the end
            signals_tail[row.sweep_number]=signal_tail #append it to the list
            curr_sstail= np.mean(signal_tail[len(signal_tail)-int(fs/10):]) #take ss of the current as last 100ms of the current step
            ih_ampltail=curr_starttail-curr_sstail
            
            #plot the Ih current in the first subplot with
            axs[1].plot(signal_tailt, signal_tail, color='gray') #plot the signal
            axs[1].plot(signal_tailt[0],signal_tail[0],'x', color='b') #plot the curr_start
            axs[1].plot(signal_tailt[-int(fs/5)],signal_tail[-int(fs/10)],'x', color='c') #plot the curr_ss
            axs[1].set_title('Ih tail')
            #fit in the curve
            if ih_ampltail > 15:
                pars_tail=pd.DataFrame(columns = ['k0','k', 'rSquared', 'tau'])
                for k0tail in range (1,31,2):
                    p0tail = (ih_ampltail, k0tail, curr_sstail) # start with values near those we expect
                    params_tail,cv_tail = sp.optimize.curve_fit(monoExp, signal_tailt, signal_tail, p0tail ,maxfev=100000)
                    m_tail,k_tail,b_tail = params_tail
                    #check goodness of fit
                    squaredDiffs_tail= np.square(signal_tail - monoExp(signal_tailt, m_tail, k_tail, b_tail))
                    squaredDiffsFromMean_tail = np.square(signal_tail - np.mean(signal_tail))
                    rSquared_tail = 1 - np.sum(squaredDiffs_tail) / np.sum(squaredDiffsFromMean_tail)
                    pars_tail= pars_tail.append({'k0':k0tail,'k':k_tail, 'rSquared': rSquared_tail, 'tau': (1 / k_tail)*1000},ignore_index=True)
                    if max(pars_tail.rSquared) > 0.9:
                        result_tail = pars_tail[pars_tail.rSquared == max(pars_tail.rSquared)].reset_index()
                        p0tail = (ih_ampltail, result_tail.k0[0], curr_sstail) # start with values near those we expect
                        params_tail,cv_tail = sp.optimize.curve_fit(monoExp, signal_tailt, signal_tail, p0tail ,maxfev=100000)
                        m_tail,k_tail,b_tail = params_tail
                        #check goodness of fit
                        squaredDiffs_tail= np.square(signal_tail - monoExp(signal_tailt, m_tail, k_tail, b_tail))
                        squaredDiffsFromMean_tail = np.square(signal_tail - np.mean(signal_tail))
                        rSquared_tail = 1 - np.sum(squaredDiffs_tail) / np.sum(squaredDiffsFromMean_tail)
                        taumsectail =( 1 / k_tail)*1000
                        axs[1].text(0.7, 0.7, f'R² = {rSquared_tail:.2f}', ha='right', va='top',style='italic',color='k', transform=axs[1].transAxes)
                        axs[1].plot(signal_tailt, monoExp(signal_tailt, m_tail, k_tail, b_tail), '--', color='r')
                        ih_tail_comment = 'good fit'
                    if 0.7 < max(pars_tail.rSquared) < 0.9:
                        result_tail = pars_tail[pars_tail.rSquared == max(pars_tail.rSquared)].reset_index()
                        p0tail = (ih_ampltail, result_tail.k0[0], curr_sstail) # start with values near those we expect
                        params_tail,cv_tail = sp.optimize.curve_fit(monoExp, signal_tailt, signal_tail, p0tail ,maxfev=100000)
                        m_tail,k_tail,b_tail = params_tail
                        #check goodness of fit
                        squaredDiffs_tail= np.square(signal_tail - monoExp(signal_tailt, m_tail, k_tail, b_tail))
                        squaredDiffsFromMean_tail = np.square(signal_tail - np.mean(signal_tail))
                        rSquared_tail = 1 - np.sum(squaredDiffs_tail) / np.sum(squaredDiffsFromMean_tail)
                        taumsectail =( 1 / k_tail)*1000
                        axs[1].text(0.7, 0.7, f'R² = {rSquared_tail:.2f} !!! lower than 0.95', ha='right', va='top',style='italic',color='r', transform=axs[1].transAxes)
                        axs[1].plot(signal_tailt, monoExp(signal_tailt, m_tail, k_tail, b_tail), '--', color='r')
                        ih_tail_comment = 'poor fit'
                    if 0.7 > max(pars_tail.rSquared):
                        rSquared_tail = np.nan
                        taumsectail = np.nan   
                        ih_tail_comment = 'no fit'
            else:
                curr_starttail = curr_sstail = ih_ampltail = taumsectail = rSquared_tail = np.nan
                axs[1].text(0.7, 0.7, 'no current', ha='right', va='top',style='italic',color='r', transform=axs[1].transAxes)
                ih_tail_comment = 'no current'

        output=output.append({'cell':file[:-4], 'sweep': row.sweep_number, 'condition':row.condition, 'compensation':row.compensation,'voltage_step':voltage_step,
                              'Ih_start': curr_start,'Ih_end': curr_ss, "Ih_ampl": ih_ampl, 'Ih_tau': taumsec,'r2offit':rSquared, 'Ih_comment': ih_comment,
                              'Ih_tail_start':curr_starttail, 'Ih_tail_end':curr_sstail, 'Ih_tail_ampl':ih_ampltail, 'Ih_tail_tau':taumsectail, 'r2offit_tail':rSquared_tail,'Ih_tail_comment':ih_tail_comment },ignore_index=True)

    except Exception as e:
        print(f"-----------------------Error processing sweep number {row.sweep_number}: {e}------------------------------")   
        
del row, index, sweep, fs, nyq, timepoints, i,v,t,art,voltage_step
del ih_start, ih_end, curr_start, currmaxindex, signal, signalt, curr_ss, ih_ampl, pars, k0,p0, params, cv, m,k,b,squaredDiffs, squaredDiffsFromMean, rSquared, result, taumsec, ih_comment
del tail_start, tail_end, currmaxindextail, curr_starttail, signal_tail, signal_tailt, curr_sstail, ih_ampltail, pars_tail, k0tail, p0tail, params_tail, cv_tail, m_tail, k_tail, b_tail, squaredDiffs_tail, squaredDiffsFromMean_tail
del rSquared_tail, result_tail, taumsectail, ih_tail_comment

#%%
output['Ih_ampl_normalized'] = output.groupby(['condition', 'compensation'])['Ih_ampl'].transform(lambda x: x / x.max())
output['Ih_tail_ampl_normalized'] = output.groupby(['condition', 'compensation'])['Ih_tail_ampl'].transform(lambda x: x / x.max())
plot_ih_steps_and_tails(signals, signals_tail, output, savedir)
#update output with correction for reasons for bad fit
output.to_csv(savedir+file[:-4]+'_ihdata.csv')
sweeps.to_csv(savedir+file[:-4]+'_sweeps_metadata.csv')
del signals, signals_tail
#%%
tables = [f for f in os.listdir(savedir) if f.endswith('.csv') and '_ihdata' in f]
        
all_results=pd.DataFrame()
for f in tables:
    temp=pd.read_csv(savedir+'/'+f)
    all_results = pd.concat([all_results,temp], axis=0, ignore_index = True)

all_results['species'] = all_results['cell'].str[0]


to_plot_comp=all_results[(all_results.species == 'H')&(all_results.compensation == 'yes')].reset_index()
to_plot_ncomp=all_results[(all_results.species == 'H')&(all_results.compensation == 'no')]

to_plot_comp=output[output.compensation == 'yes']
to_plot_ncomp=output[output.compensation == 'no']


plt.figure()
sns.lineplot(data=to_plot_comp, x='voltage_step', y='Ih_ampl_normalized', hue='condition')
sns.lineplot(data=to_plot_comp, x='voltage_step', y='Ih_tail_ampl_normalized', hue='condition')
plt.title('comp')


plt.figure()
sns.lineplot(data=to_plot_ncomp, x='voltage_step', y='Ih_ampl_normalized', hue='condition')
sns.lineplot(data=to_plot_ncomp, x='voltage_step', y='Ih_tail_ampl_normalized', hue='condition')
plt.title('ncomp')

plt.figure()
sns.lineplot(data=to_plot_comp, x='voltage_step', y='Ih_ampl_normalized', hue='condition')
sns.lineplot(data=to_plot_comp, x='voltage_step', y='Ih_tail_ampl_normalized', hue='condition')
plt.title('comp_norm')


plt.figure()
sns.lineplot(data=to_plot_ncomp, x='voltage_step', y='Ih_ampl_normalized', hue='condition')
sns.lineplot(data=to_plot_ncomp, x='voltage_step', y='Ih_tail_ampl_normalized', hue='condition')
plt.title('ncomp_norm')


plt.figure()
sns.lineplot(x="voltage_step", y='Ih_tail_ampl_normalized',data=all_results[all_results.r2offit_tail > 0.9], hue='condition',hue_order=['baseline', 'naag'], palette=['blue', 'orange'], markers=['o', 'o'], dashes=False, err_style="bars", ci=68, linestyle='--', err_kws={'capsize':5, 'capthick':2})

trial=all_results[all_results.r2offit_tail > 0.9]

import scipy.optimize as opt
def boltz_act(x, Vhalf, k):
    return 1 / (1+np.exp((Vhalf-x)/k))
def boltz_inact(x, Vhalf, k):
    return 1 / (1+np.exp(-(Vhalf-x)/k))

tmp=trial.loc[(~trial.Ih_ampl_normalized.isna())]
popt, _ = opt.curve_fit(boltz_act, tmp.voltage_step, tmp.Ih_ampl_normalized)
y = boltz_act(x, popt[0], popt[1])
plt.plot(x, y, 'k')
#%%
tables = [f for f in os.listdir('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Ih_current/analysed_data/trial') if f.endswith('.csv') and '_ihdata' in f]

all_results=pd.DataFrame()
for f in tables:
    temp=pd.read_csv('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/Ih_current/analysed_data/trial'+'/'+f)
    all_results = pd.concat([all_results,temp], axis=0, ignore_index = True)

all_results['species'] = all_results['cell'].str[0]

t=all_results[(all_results.species == 'H')&(all_results.compensation == 'yes')]
tn=all_results[(all_results.species == 'M')&(all_results.compensation == 'no')]



plt.figure()
sns.lineplot(data=t, x='voltage_step', y='Ih_ampl', hue='condition')
sns.lineplot(data=t, x='voltage_step', y='Ih_tail_ampl', hue='condition')
plt.title('human comp')

plt.figure()
sns.lineplot(data=tn, x='voltage_step', y='Ih_ampl', hue='condition')
sns.lineplot(data=tn, x='voltage_step', y='Ih_tail_ampl', hue='condition')
plt.title('mouse comp')


plt.figure()
sns.lineplot(data=t, x='voltage_step', y='Ih_ampl_normalized', hue='condition')
sns.lineplot(data=t, x='voltage_step', y='Ih_tail_ampl_normalized', hue='condition')
plt.title('comp')


plt.figure()
sns.lineplot(data=tn, x='voltage_step', y='Ih_ampl_normalized', hue='condition')
sns.lineplot(data=tn, x='voltage_step', y='Ih_tail_ampl_normalized', hue='condition')
plt.title('ncomp')



condition_palette = {'baseline': '#009FE3','naag': '#F3911A'}

data=all_results[all_results.compensation == 'yes']

plt.figure()
sns.lineplot(data=data, x='voltage_step', y='Ih_ampl_normalized',palette=condition_palette, 
             hue='condition', style='species', ci='sd')
sns.lineplot(data=data, x='voltage_step', y='Ih_tail_ampl_normalized',palette=condition_palette, 
             hue='condition', style='species', ci='sd')
plt.legend(title='Legend Title', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('comp')





# if timepoints[0] == 292 and len (timepoints) == 6:
        #     i, v, t = butter_lowpass_filter(data=sweep.i, cutoff=500, fs=fs, order=3), sweep.v, sweep.t
        #     art = 150
        # elif timepoints[0] == 292 and len (timepoints) != 6:
        #     temp=dataset.sweep(sweep_number=row.sweep_number-1)
        #     timepoints = (np.where(temp.v[:-1] != temp.v[1:])[0]) + 1
        # elif timepoints[0] != 292 and len (timepoints) == 6:
        #     diff=timepoints[0]-292
        #     i, v, t = butter_lowpass_filter(data=sweep.i[diff*2:], cutoff=500, fs=fs, order=3), sweep.v[diff*2:], sweep.t[diff*2:]
        #     art = 250
        #     timepoints = (np.where(v[:-1] != v[1:])[0]) + 1
        # elif timepoints[0] != 292 and len (timepoints) != 6:   
        #     diff=timepoints[0]-292
        #     i, v, t = butter_lowpass_filter(data=sweep.i[diff*2:], cutoff=500, fs=fs, order=3), sweep.v[diff*2:], sweep.t[diff*2:]
        #     art = 250
        #     temp=dataset.sweep(sweep_number=row.sweep_number-1)
