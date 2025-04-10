#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:22:08 2023

@author: annagalakhova

Analysis script for the data recorded with extracellular stimulation electrode
Functions are stored in toolkit_ag.py
Sweeps cleaning is performed in QC_check.py manually
"""
#%%importing the modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipfx.dataset.create import create_ephys_data_set
from ipfx.feature_extractor import SpikeFeatureExtractor
import scipy as sp
from scipy.stats import zscore
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.signal import order_filter
from scipy.signal import firwin
from scipy.signal import lfilter
import seaborn as sns
from ipfx.utilities import drop_failed_sweeps
from ipfx.data_set_features import extract_data_set_features
import os
from datetime import datetime, timedelta
current_date=datetime.now().date()
import statsmodels.api as sm
from statsmodels.formula.api import ols
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import h5py
import pingouin as pg
from scipy import stats
import math
#%%QC fucntions 
def rm_check2(file,condition, sweeps_cleared, dataset,exp=1):
    QCheck = pd.DataFrame(columns = ['cell','sweep_number', 'condition', 'leak', 'Rm', 'Outlier?'])
    rm=[]
    sweep_nrs = []
    leak=[]
    df = sweeps_cleared[(sweeps_cleared.stimulus_code.str.contains(condition))& (sweeps_cleared.clamp_mode.values == 'CurrentClamp')].reset_index(drop=True)
    if df.shape[0] != 0:
    #check the Ra
        for i, number2 in enumerate(df.sweep_number):
            sweep = dataset.sweep(sweep_number=number2)
            v=sweep.v
            tp_volt = min(v[300:1500])
            tp_ampl = v[300]-tp_volt
            rm_momh = tp_ampl/50*1000
            sweep_nrs.append(number2)
            rm.append(rm_momh)
            leak.append(df.leak_pa[i])
        QCheck['sweep_number'] = sweep_nrs
        if exp==1:
            if 'ba' in condition:
                QCheck['condition'] = 'baseline'
            elif 'na' in condition:
                QCheck['condition'] = 'naag'
            elif 'wa' in condition: 
                QCheck['condition'] = 'washout'
        elif exp==2:
            QCheck['condition']=df.condition[number2]
        QCheck['leak'] = leak
        QCheck['Rm']=rm
        if len(rm)>3:
            rm_normality = pg.normality(rm)
            rm_mean = np.mean(rm)
            rm_std = np.std(rm)
            z_scores = stats.zscore(rm)
            outlier = []
            for i in z_scores:
                if abs(i)>2:
                    outlier.append('true')
                else:
                    outlier.append('false')
            QCheck['Outlier?'] = outlier
            QCheck['cell'] = file
        else:
            QCheck['Outlier?'] = 'nan'
            QCheck['cell'] = file
    else:
        QCheck= QCheck.append({'cell':file,'sweep_number': 'no sweeps','condition': condition, 'leak':'nan', 'Rm':'nan', 'Outlier?':'nan'}, ignore_index=True)
    return QCheck

def rm_check3(file,sweeps_cleared, dataset):
    QCheck = pd.DataFrame(columns = ['cell','sweep_number', 'condition', 'leak', 'Rm', 'Outlier?'])
    rm=[]
    sweep_nrs = []
    leak=[]
    conds=[]
    df = sweeps_cleared[sweeps_cleared.clamp_mode.values == 'CurrentClamp']
    if df.shape[0] != 0:
    #check the Ra
        for index, row in df.iterrows():
            sweep = dataset.sweep(sweep_number=index)
            v=sweep.v
            tp_start, tp_end = sweep.epochs['test'][0],sweep.epochs['test'][1]
            tp_volt = min(v[tp_start:tp_end])
            tp_ampl = np.mean(v[tp_start:tp_start+100])-tp_volt
            rm_momh = tp_ampl/50*1000
            sweep_nrs.append(row.sweep_number)
            rm.append(rm_momh)
            leak.append(row.leak_pa)
            conds.append(row.condition)
        QCheck['sweep_number'] = sweep_nrs
        QCheck['condition']=conds
        QCheck['leak'] = leak
        QCheck['Rm']=rm
        if len(rm)>3:
            rm_normality = pg.normality(rm)
            rm_mean = np.mean(rm)
            rm_std = np.std(rm)
            z_scores = stats.zscore(rm)
            outlier = []
            for i in z_scores:
                if abs(i)>2:
                    outlier.append('true')
                else:
                    outlier.append('false')
            QCheck['Outlier?'] = outlier
            QCheck['cell'] = file
        else:
            QCheck['Outlier?'] = 'nan'
            QCheck['cell'] = file
    else:
        QCheck= QCheck.append({'cell':file,'sweep_number': 'no sweeps','condition': 'nan', 'leak':'nan', 'Rm':'nan', 'Outlier?':'nan'}, ignore_index=True)
    return QCheck

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b
#exp is experiment - for analysis of experiment 1 or 2
def ra_rin_check(file, sweeps_cleared,dataset, exp=1):
    output = pd.DataFrame(columns = ['Ra','Rin', 'leak', 'sweep_number', 'condition', 'cell', 'tau(ms)', 'sampling_rate'])
    df = sweeps_cleared[(sweeps_cleared.stimulus_code.str.contains('ulse')) & (sweeps_cleared.clamp_mode.values == 'VoltageClamp')].reset_index(drop=True)
    if df.shape[0] != 0:
        if exp==1:
            for number3, r in enumerate(df.sweep_number):
                sweep = dataset.sweep(sweep_number=r)
                fs=sweep.sampling_rate
                stim_index = np.nonzero(sweep.v)[0][0]
                signal=sweep.v[stim_index:stim_index+2500]
                if len(np.where(signal > 10)[0]) > 0:
                    last_index = np.where(signal > 10)[0][-1]+stim_index
                else:
                    continue
                i=sweep.i[0:last_index]
                v=sweep.v[0:last_index]
                bsl = i[stim_index-200]
                peak = max(i)
                peak_ind=np.argmax(i)
                ss = i[last_index-1]
                delta = peak-bsl
                delta_rin = ss-bsl
                ra_i = (v[stim_index]/delta)*1000
                rin_i = (v[stim_index]/delta_rin)*1000
                y=i[peak_ind:last_index]
                x = np.arange(0, len(y))
                m_guess = 500
                b_guess = -100
                t_guess = 0.03
                (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess])
                plt.figure()
                plt.plot(x, monoExp(x, m_true, t_true, b_true), '--')
                plt.plot(x, y, )
                plt.title(r)
                tau=-1/(-t_true)/fs*1000
                if 'ba' in df.stimulus_code[number3]:
                    condition='baseline'
                elif 'na' in df.stimulus_code[number3]:
                    condition='naag'
                else:
                    condition='washout'
                output= output.append({'Ra':ra_i, 'Rin':rin_i, 'leak':bsl, 'sweep_number' : r, 'condition': condition, 
                                   'cell':file, 'tau(ms)':tau, 'sampling_rate':fs}, ignore_index=True)
        elif exp == 2:
            for number3, r in enumerate(df.sweep_number):
                sweep = dataset.sweep(sweep_number=r)
                fs=sweep.sampling_rate
                stim_index = np.nonzero(sweep.v)[0][0]
                signal=sweep.v[stim_index:stim_index+2500]
                if len(np.where(signal > 10)[0]) > 0:
                    last_index = np.where(signal > 10)[0][-1]+stim_index
                else:
                    continue
                i=sweep.i[0:last_index]
                v=sweep.v[0:last_index]
                bsl = i[stim_index-200]
                peak = max(i)
                peak_ind=np.argmax(i)
                ss = i[last_index-1]
                delta = peak-bsl
                delta_rin = ss-bsl
                ra_i = (v[stim_index]/delta)*1000
                rin_i = (v[stim_index]/delta_rin)*1000
                y=i[peak_ind:last_index]
                x = np.arange(0, len(y))
                m_guess = 500
                b_guess = -100
                t_guess = 0.03
                (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp, x, y, [m_guess, t_guess, b_guess])
                plt.figure()
                plt.plot(x, monoExp(x, m_true, t_true, b_true), '--')
                plt.plot(x, y, )
                plt.title(r)
                tau=-1/(-t_true)/fs*1000
                condition=df.condition[number3]
                output= output.append({'Ra':ra_i, 'Rin':rin_i, 'leak':bsl, 'sweep_number' : r, 'condition': condition, 
                                   'cell':file, 'tau(ms)':tau, 'sampling_rate':fs}, ignore_index=True)
            
    else:
        output= output.append({'Ra':'nan','Rin':'nan','leak':'nan', 'sweep_number' : 'nan', 'condition': 'nan', 
                           'cell':file, 'tau(ms)':'nan', 'sampling_rate':'nan'}, ignore_index=True)
    beg=0
    normra = output.Ra[beg]
    normrin = output.Rin[beg]
    normtau=output['tau(ms)'][beg]
    output['norm_Ra'] = output['Ra'] / normra
    output['norm_Rin'] = output['Rin'] / normrin
    output['norm_tau'] = output['tau(ms)'] / normtau
    output=output[beg:]
    return output

def close():
    plt.close('all')
#%%trace processing - filtering, creating averages
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
    
def butter_lowpass_filter(dataset, data, cutoff, fs, order):
    sweep = dataset.sweep(sweep_number=0)
    fs = sweep.sampling_rate
    cutoff = 120
    nyq = 0.5*fs
    order = 4
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#function to create an average of the trace and it's filtering to filter out the artefact
def artefact_filtering(dataset,trace,show_plot=True):
    sweep=dataset.sweep(sweep_number= 0)
    artifact_duration = 0.002  # Artifact duration in seconds
    sampling_frequency = sweep.sampling_rate  # Sampling frequency in Hz
    cutoff_frequency = 1500  # Cutoff frequency in Hz
    # Calculate the filter length (number of taps)
    filter_length = int(artifact_duration * sampling_frequency) * 2 + 1
    # Calculate the normalized cutoff frequency
    normalized_cutoff = cutoff_frequency / (sampling_frequency / 2)
    # Design the FIR filter using a Hamming window
    fir_filter = firwin(filter_length, normalized_cutoff, window='hamming')
    # Apply the FIR filter forward to the data (avg_baseline1 is your data array)
    filtered_data_forward = lfilter(fir_filter, [1.0], trace)
    # Reverse the data
    reversed_data = np.flip(filtered_data_forward)
    # Apply the FIR filter backward to the reversed data
    filtered_data_backward = lfilter(fir_filter, [1.0], reversed_data)
    # Reverse the result back to the original order to get the zero-phase filtered data
    filtered_data_zero_phase = np.flip(filtered_data_backward)
    margin=int(sampling_frequency/250)
    filtered_data_zero_phase[0:margin] = np.mean(filtered_data_zero_phase[margin:margin+100])
    filtered_data_zero_phase[-margin:] = np.mean(filtered_data_zero_phase[-margin*2:-margin+1])
    if show_plot:
        plt.figure()
        plt.plot(trace)
        plt.plot(filtered_data_zero_phase)
        plt.title('filtering results')
    return filtered_data_zero_phase

#function creating an average: mode=0 with no filtering, mode=1 with filtering, default mode=1
def get_avg(dataset,sweeps,sweepset,clamp_mode,set_nr=None,down=0, mode=1):
    df = sweeps.loc[(sweeps['stimulus_code'].str.contains(sweepset)) & (sweeps.clamp_mode.values == clamp_mode)]
    if set_nr is not None and 'set_number' in df.columns:
        df = df[df['set_number'] == set_nr]

    swps=[]
    for i in df.sweep_number:
        sweep=dataset.sweep(sweep_number=i)
        if down ==0:
            final_sweepv = sweep.v
        elif down ==1:
            try:
                final_sweepv=downsample(dataset,i)
            except ValueError as e:  # Catch the ValueError and store it in variable 'e'
            # If the downsample function doesn't work due to low sampling frequency,
            # set final_sweepv to sweep.v and print the error message
                print(f"Downsample failed: {str(e)}")
                final_sweepv = sweep.v
        if clamp_mode == 'CurrentClamp':
            swps.append(final_sweepv)
        elif clamp_mode == 'VoltageClamp':
            swps.append(sweep.i)
    avg=np.mean(swps,axis=0)
    if mode == 0:
        return avg  # Return the unfiltered average
    elif mode == 1:
        filt = artefact_filtering(dataset,avg)
        plt.figure()
        plt.plot(filt)
        plt.title('Average (filtered)')
        return filt  # Return the filtered average
    else:
        raise ValueError("Invalid mode value. Use 0 for no filter or 1 for filtering.")
   


#%%stimset-specific analysis
#patchseq protocols analysis - subthreshold and rheobase
def calc_ps_rheo(file,dataset, sweeps, folder):
    df = sweeps[(sweeps.stimulus_code.str.contains('Rheo')) & (sweeps.clamp_mode.values == 'CurrentClamp')].reset_index(drop=True)
    df.drop_duplicates(subset='stimulus_code_ext', keep='first', inplace=True)
    if df.shape[0] != 0:
        drop_failed_sweeps(dataset)
        ps_sweeps = dataset.filtered_sweep_table()
        cell_features, sweep_features, cell_record, sweep_records, _, _ = \
            extract_data_set_features(dataset, subthresh_min_amp=-100.0)
        df = ps_sweeps[(ps_sweeps.stimulus_code.str.contains('Rheo')) & (ps_sweeps.clamp_mode.values == 'CurrentClamp')]
        df = df.reset_index(drop=True)
        df.drop_duplicates(subset='stimulus_code_ext', keep='first', inplace=True)
        if df.shape[0] != 0:
            output=pd.DataFrame(columns = ['cell','condition','rheobase', 'rheo_fth','rheo_fupstroke', 'rheo_fap_v', 'rheo_fap_width'])
            rheos_b=[]
            rheos_n=[]
            swpsb=[]
            swpsn=[]
            for index, row in df.iterrows():
                if len(sweep_features[row.sweep_number]['spikes']) != 0:
                    irheo= round(sweep_features[row.sweep_number]['spikes'][0]['threshold_i'])
                    condition=row.stimulus_code[5:]
                    if 'ba' in condition:
                        rheos_b.append(irheo)
                        swpsb.append(row.sweep_number)
                    elif 'na' in condition:
                        rheos_n.append(irheo)
                        swpsn.append(row.sweep_number)
                else:
                    continue
            rheobb =min(rheos_b)
            rheobn =min(rheos_n)
            rheoindb=rheos_b.index(rheobb)
            rheoindn=rheos_n.index(rheobn)
            output=output.append({'cell':file,'condition':'baseline','rheobase':rheobb, 'rheo_fth':sweep_features[swpsb[rheoindb]]['spikes'][0]['threshold_v'],
                                  'rheo_fupstroke':sweep_features[swpsb[rheoindb]]['spikes'][0]['upstroke'], 'rheo_fap_v':sweep_features[swpsb[rheoindb]]['spikes'][0]['peak_v'], 'rheo_fap_width':sweep_features[swpsb[rheoindb]]['spikes'][0]['width']}, ignore_index=True) 
            output=output.append({'cell':file,'condition':'naag','rheobase':rheobn, 'rheo_fth':sweep_features[swpsn[rheoindn]]['spikes'][0]['threshold_v'],
                                  'rheo_fupstroke':sweep_features[swpsn[rheoindn]]['spikes'][0]['upstroke'], 'rheo_fap_v':sweep_features[swpsn[rheoindn]]['spikes'][0]['peak_v'], 'rheo_fap_width':sweep_features[swpsn[rheoindn]]['spikes'][0]['width']}, ignore_index=True)
    else:
        print('rheo did not pass')
        output = None 
    plt.figure()
    plt.plot(dataset.sweep(sweep_number=swpsb[rheoindb]).v)
    plt.plot(dataset.sweep(sweep_number=swpsn[rheoindn]).v)
    plt.savefig(folder+'/rheo'+'.eps', format='eps')
    return output
    
    
#function to calculate the resonance frequency and 
def res_freq(dataset,sweeps,file,folder):
    #get the data from the dataset file
    df = sweeps[sweeps.stimulus_code.str.contains('CHIRP')].reset_index(drop=True)
    if df.shape[0] != 0:
        chirp_sweeps_b = []
        chirp_stims_b = []
        chirp_times_b = []
        chirp_sweeps_n = []
        for index, row in df.iterrows():
            sweep=dataset.sweep(sweep_number=row.sweep_number)
            v=sweep.v
            i=sweep.i
            t=sweep.t
            condition=row.stimulus_code[6:]
            if len(v) == 590000 and max(v)<-45:
                if 'ba' in condition:
                    chirp_sweeps_b.append(v)
                    chirp_stims_b.append(i)
                    chirp_times_b.append(t)
                elif 'na' in condition:
                    chirp_sweeps_n.append(v)
            elif len(v) != 590000:
                v=downsample(dataset,row.sweep_number)
                i=downsample(dataset,row.sweep_number, mode='i')
                t=downsample(dataset,row.sweep_number, mode='t')
                if 'ba' in condition:
                    chirp_sweeps_b.append(v)
                    chirp_stims_b.append(i)
                    chirp_times_b.append(t)
                elif 'na' in condition:
                    chirp_sweeps_n.append(v)
        #process the averaged data
        avg_baseline = np.mean(chirp_sweeps_b, axis=0)
        avg_naag=np.mean(chirp_sweeps_n, axis=0)
        stim = np.mean(chirp_stims_b, axis=0)
        t=np.mean(chirp_times_b, axis=0)
        traces={'baseline': avg_baseline, 'naag':avg_naag}
        output=pd.DataFrame(columns = ['cell','condition', 'f_res', '3db_cutoff'])
        plt.figure()
        plt.plot(t,traces['baseline'])
        plt.plot(t,traces['naag'])
        plt.title(file)
        plt.savefig(folder+'/res_'+file+'.eps', format='eps')
        plt.figure()
        for k in traces.keys():
            avg=traces[k]
            #filter the data
            signal=artefact_filtering(dataset,avg, show_plot=False)
            #plot the data
            L = len(signal)
            fs = sweep.sampling_rate
            if fs != 25000.0:
                    fs=25000.0
            f = np.arange(0, L) * fs / L
            # plt.figure()
            # plt.plot(t,signal)
            # plt.xlim([0.6,20.6])
            # plt.ylabel('Voltage (mV)', fontsize=12)
            # plt.xlabel('Time (sec)', fontsize=12)
            # plt.title(k)
            # plt.show()
            #fast Fourier transformation
            signal_fft = np.fft.fft(signal)
            stim_fft = np.fft.fft(stim)
            y=abs(signal_fft) / abs(stim_fft) * 1000
            #look onlt at 0.7-40 Hz
            xmin = 0.7
            xmax = 40
            mask = (f >= xmin) & (f <= xmax)
            zoomed_f = f[mask]
            zoomed_y = y[mask]
            cutoff_freq = 4000 
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
            plt.title(file)
            plt.savefig(folder+'/res_imp'+file+'.eps', format='eps')
            output=output.append({'cell':file,'condition':k, 'f_res':f_res, '3db_cutoff': cutoff_3db}, ignore_index=True)
    else:
        raise ValueError("no data found")
    return output
    
#CCsteps analysis - excitability
def excitability(file,dataset,sweeps, folder):
    df = sweeps[(sweeps.stimulus_code.str.contains('CCsteps')) & (sweeps.clamp_mode.values == 'CurrentClamp')].reset_index(drop=True)
    #df=df.drop_duplicates(subset='stimulus_code_ext', keep='first')
    output_cl = pd.DataFrame()
    output = pd.DataFrame()
    graph_cl = pd.DataFrame(columns=['cell','condition', 'current', 'clamp_mode', 'nopaps', 'first_upstroke','first_downstroke','first_threshold', 'first_ap_v','first_width','first_updownratio','sweep_nr'])
    graph = pd.DataFrame(columns=['cell','condition','current', 'clamp_mode', 'nopaps',  'cell', 'first_upstroke','first_downstroke','first_threshold','first_ap_v','first_width','first_updownratio','sweep_nr'])
    if df.shape[0] != 0 and len(df.stimulus_code.unique())>1:
        for index, row in df.iterrows():
            sweep = dataset.sweep(sweep_number=row.sweep_number)
            if sweep.epochs['stim'] is not None:
                start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                ext = SpikeFeatureExtractor()
                results = ext.process(sweep.t, sweep.v, sweep.i)
                if len(results)>0:
                    cond=(results['peak_index'] >= start_idx) & (results['peak_index'] <= end_idx)
                    results['sweep_number'] = row.sweep_number
                    if 'CL' in row.stimulus_code:
                        results['clamp_mode']='clamped'
                        if 'ba' in row.stimulus_code:
                            condition='baseline'
                        elif 'na' in  row.stimulus_code:
                            condition='naag'
                        elif 'wa' in row.stimulus_code:
                            condition='washout'
                        results['condition']=condition
                        results['cell']=file
                        output_cl = pd.concat([output_cl, results], axis = 0)
                        graph_cl=graph_cl.append({'cell':file,'condition':condition,'current':round(max(sweep.i[start_idx:end_idx],key=abs)),'clamp_mode':'clamped', 'nopaps':len(results[cond]), 'first_upstroke':results['upstroke'].iloc[0],'first_downstroke':results['downstroke'].iloc[0],'first_threshold':results['threshold_v'].iloc[0],'first_ap_v':results['peak_v'].iloc[0],'first_width':results['width'].iloc[0],'first_updownratio':results['upstroke_downstroke_ratio'].iloc[0], 'sweep_nr':results['sweep_number'].iloc[0]}, ignore_index=True)
                    else:
                        results['clamp_mode']='not_clamped'
                        if 'ba' in row.stimulus_code:
                            condition='baseline'
                        elif 'na' in  row.stimulus_code:
                            condition='naag'
                        elif 'wa' in row.stimulus_code:
                            condition='washout'
                        results['condition']=condition
                        results['cell']=file
                        output = pd.concat([output, results], axis = 0)
                        graph=graph.append({'cell':file,'condition':condition,'current':round(max(sweep.i[start_idx:end_idx],key=abs)),'clamp_mode':'not_clamped', 'nopaps':len(results[cond]), 'first_upstroke':results['upstroke'].iloc[0],'first_downstroke':results['downstroke'].iloc[0],'first_threshold':results['threshold_v'].iloc[0],'first_ap_v':results['peak_v'].iloc[0],'first_width':results['width'].iloc[0],'first_updownratio':results['upstroke_downstroke_ratio'].iloc[0], 'sweep_nr':results['sweep_number'].iloc[0]}, ignore_index=True)
    else:
        return
    if len(graph) != 0:
        plt.figure()
        sns.lineplot(x='current', y='nopaps', hue='condition', data=graph)
        plt.title(file+'_'+graph.clamp_mode[0], fontsize=8)
        plt.savefig(folder+'/steps_'+file+'.eps', format='eps')
    if len(graph_cl) != 0:    
        plt.figure()
        sns.lineplot(x='current', y='nopaps', hue='condition', data=graph_cl)
        plt.title(file+'_'+graph_cl.clamp_mode[0], fontsize=8)
        plt.savefig(folder+'/steps_'+file+'.eps', format='eps')
    outputd={'spikes_clamped':output_cl, 'graph_clamped':graph_cl,'spikes_not_clamped':output,'graph_not_clamped':graph}
    return outputd

#function to calculate passive properties and sag_ratio
def passive_features(file,dataset, sweeps, folder):
    df = sweeps[(sweeps.stimulus_code.str.contains('TestPulse')) & (sweeps.clamp_mode.values == 'VoltageClamp')]
    df = df.reset_index(drop=True)
    df2 = sweeps[(sweeps.stimulus_code.str.contains('rmp')) & (sweeps.clamp_mode.values == 'CurrentClamp')]
    df2 = df2.reset_index(drop=True)
    output=pd.DataFrame(columns = ['sweep_number','condition','ra', 'leak', 'tau(ms)'])
    if df.shape[0] != 0:
        for index, row in df.iterrows():
            #select the testpulse part of your sweep (epoch 'test')
            sweep = dataset.sweep(sweep_number=row.sweep_number)
            sweep_number=sweep.sweep_number
            start_index, end_index = sweep.epochs['test'][0],sweep.epochs['test'][1]
            fs=sweep.sampling_rate
            i=sweep.i[start_index:end_index]
            v = sweep.v[start_index:end_index]
            stim_index = np.nonzero(v)[0][0]
            if len(np.where(abs(v) > 10)[0]) > 0:
                last_index = np.where(abs(v) > 10)[0][-1]
            else:
                continue
            if v[last_index]<0:
                i=-i
            #calculate Ra, leak
            bsl = np.mean(i[stim_index-200:stim_index-100], axis=0) #leak
            peak_id,peak = np.argmax(i),max(i)
            delta = peak-bsl
            ra_i = (v[stim_index]/delta)*1000
            #calculate tau
            y=i[peak_id:last_index]
            x = np.arange(0, len(y))
            m_guess = 500
            b_guess = -100
            t_guess = 0.03
            if len(y) !=0:
                (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp_cc, x, y, [m_guess, t_guess, b_guess])
                tau=-1/(-t_true)/fs*1000
                if 'ba' in row.stimulus_code:
                    condition = 'baseline'
                elif 'na' in row.stimulus_code:
                    condition='naag'
                elif 'wa' in row.stimulus_code:
                    condition = 'washout'
                plt.figure()
                plt.plot(x, monoExp_cc(x, m_true, t_true, b_true), '--')
                plt.plot(x, y, )
                plt.title(str(row.sweep_number)+'_'+str(condition))
            else: 
                continue
            # plt.savefig(folder+'/tau_'+str(row.sweep_number)+'_'+str(condition)+'.eps', format='eps')
            #ss = i[last_index-2]
            #delta_rin = ss-bsl
            #rin_i = (v[stim_index]/delta_rin)*1000
            if v[last_index]<0:
                output=output.append({'sweep_number':sweep_number,'condition':condition,'ra':-ra_i, 'leak':-bsl, 'tau(ms)':tau}, ignore_index=True)
            else:
                output=output.append({'sweep_number':sweep_number,'condition':condition,'ra':ra_i, 'leak':bsl, 'tau(ms)':tau}, ignore_index=True)
    else: 
        raise ValueError("no data found")
    rmps={}
    if df2.shape[0] != 0:
        for index2, row2 in df2.iterrows():
            sweep = dataset.sweep(sweep_number=row2.sweep_number)
            rmp = np.mean(sweep.v[sweep.epochs['test'][1]:], axis=0)
            if 'ba' in row2.stimulus_code:
                condition = 'baseline'
            elif 'na' in row2.stimulus_code:
                condition='naag'
            elif 'wa' in row2.stimulus_code:
                condition = 'washout'
            if condition in rmps:
                continue
            else:
                rmps[condition] = rmp
    else:
        df2 = sweeps[(sweeps.stimulus_code.str.contains('CCsteps')) & (sweeps.clamp_mode.values == 'CurrentClamp')]
        df2 = df2.reset_index(drop=True)
        df2=df2.drop_duplicates(subset='stimulus_code_ext', keep='first')
        df2=df2[df2['stimulus_code_ext'].str.contains(r'\[0\]')]
        if df2.shape[0] !=0:
            for index2, row2 in df2.iterrows():
                #select the testpulse part of your sweep (epoch 'test')
                sweep = dataset.sweep(sweep_number=row2.sweep_number)
                sweep_number=sweep.sweep_number
                start_ind,end_ind= sweep.epochs['test'][1],sweep.epochs['stim'][0]
                rmp = np.mean(sweep.v[start_ind:end_ind], axis=0)
                if 'ba' in row2.stimulus_code:
                    condition = 'baseline'
                elif 'na' in row2.stimulus_code:
                    condition='naag'
                elif 'wa' in row2.stimulus_code:
                    condition = 'washout'
                if condition in rmps:
                    continue
                else:
                    rmps[condition] = rmp
        else:
            print('no rmp values')
    baseline_rows = output[output.condition == 'baseline']
    output.loc[baseline_rows.index[0], 'rmp'] = rmps.get('baseline')
    output.loc[baseline_rows.index[0], 'av_leak'] = baseline_rows['leak'].mean()
    output.loc[baseline_rows.index[0], 'av_tau'] = baseline_rows['tau(ms)'].mean()
    # Process 'naag'
    naag_rows = output[output.condition == 'naag']
    output.loc[naag_rows.index[0], 'rmp'] = rmps.get('naag')
    output.loc[naag_rows.index[0], 'av_leak'] = naag_rows['leak'].mean()
    output.loc[naag_rows.index[0], 'av_tau'] = naag_rows['tau(ms)'].mean()
    if 'washout' in rmps.keys():
        washout_rows = output[output.condition == 'washout']
        output.loc[washout_rows.index[0], 'rmp'] = rmps.get('washout')
        output.loc[washout_rows.index[0], 'av_leak'] = washout_rows['leak'].mean()
        output.loc[washout_rows.index[0], 'av_tau'] = washout_rows['tau(ms)'].mean()
    output['cell']=file
    return output

#function to calculate sag, applicable to any negative step protocol, used in ps_analysis
def calculate_sag(file, dataset,sweeps,stim='CCsteps', mode=1):
    df3 = sweeps[(sweeps.stimulus_code.str.contains(stim)) & (sweeps.clamp_mode.values == 'CurrentClamp')].reset_index(drop=True)
    #df3.drop_duplicates(subset='stimulus_code_ext', keep='first', inplace=True)
    curr_in={}
    rins={}           
    if df3.shape[0] != 0:
        sag_result=pd.DataFrame(columns = ['sweep_number','condition','clamp_mode','curr_inj', 'base','sag_volt', 'ss_volt', 'sag','sag_ratio', 'volt_defl'])
        for index3, row3 in df3.iterrows():
            sweep = dataset.sweep(sweep_number=row3.sweep_number)
            sweep_number = sweep.sweep_number
            if sweep.epochs['stim'] is not None:
                start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                dur=int(abs(start_idx-end_idx))
                input_trace = sweep.i[:end_idx]
                if mode == 1:
                    output_trace= artefact_filtering(dataset,sweep.v[:end_idx], show_plot=False)
                else:
                    output_trace = sweep.v[:end_idx]
                curr_inj = float(round(min(input_trace[start_idx:end_idx])))
                if curr_inj <0:
                    sag_volt_ind = np.where(output_trace==min(output_trace[start_idx:round(start_idx+(dur/5))]))[0][0]
                    sag_volt = np.mean(output_trace[sag_volt_ind-25:sag_volt_ind], axis=0)
                    ss_volt = np.mean(output_trace[round(end_idx-(dur/5)):end_idx], axis=0)
                    base = np.mean(output_trace[start_idx-800:start_idx-300], axis=0)
                    mini_delta=ss_volt-sag_volt
                    sag_delta=base-sag_volt
                    sag_ratio = mini_delta/sag_delta
                    rin=(ss_volt-base)/curr_inj*1000
                    if 'ba' in row3.stimulus_code:
                        condition = 'baseline'
                    elif 'na' in row3.stimulus_code:
                        condition='naag'
                    elif 'wa' in row3.stimulus_code:
                        condition = 'washout'
                    if 'CL' in row3.stimulus_code:
                        cmode='clamped'
                    else:
                        cmode='not_clamped'
                    curr_in[str(str(sweep_number)+'_'+condition)] = curr_inj
                    rins[str(str(sweep_number)+'_'+condition)] = rin
                    sag_result= sag_result.append({'sweep_number': sweep_number, 'condition':condition,'clamp_mode':cmode,
                                     'curr_inj':curr_inj, 'base':base,'sag_volt':sag_volt, 'ss_volt':ss_volt, 'sag':mini_delta,
                                     'sag_ratio':sag_ratio,'volt_defl':sag_delta}, ignore_index=True)
                    plt.figure()
                    plt.plot(output_trace)
                    plt.axhline(sag_volt, color='r')
                    plt.axhline(ss_volt, color='c')
                    plt.axhline(base, color='b')
                    plt.title("Sweep number: {}, curr_inj: {}, Condition: {}".format(sweep_number,curr_inj, condition))
                else:
                    continue
            else:
                continue
    baseline_c = []
    baseline_r = []
    naag_c = []
    naag_r = []
    w_c = []
    w_r = []
    for key, value in curr_in.items():
        if 'bas' in key:
            baseline_c.append(value)
        elif 'naag' in key:
            naag_c.append(value)
        elif 'was' in key:
            w_c.append(value)
    for key2, value2 in rins.items():
        if 'bas' in key2:
            baseline_r.append(value2)
        elif 'naag' in key2:
            naag_r.append(value2)
        elif 'was' in key2:
            w_r.append(value2)
    rin_b=np.polyfit(baseline_c,baseline_r, 1)[1]
    rin_n=np.polyfit(naag_c,naag_r, 1)[1]
    if (sag_result['condition'].str.contains('was')).any():
        rin_w=np.polyfit(w_c,w_r, 1)[1]
    baseline_rows = sag_result[sag_result.condition == 'baseline']
    sag_result.loc[baseline_rows.index[0], 'rin'] = rin_b
    naag_rows = sag_result[sag_result.condition == 'naag']
    sag_result.loc[naag_rows.index[0], 'rin'] = rin_n
    if (sag_result['condition'].str.contains('was')).any():
        was_rows = sag_result[sag_result.condition == 'washout']
        sag_result.loc[was_rows.index[0], 'rin'] = rin_w
    sag_result['cell']=file
    mask=sag_result.volt_defl > 3
    sag_result=sag_result[mask]
    return sag_result

#function to choose the most overlapping sag_volt sweeps and plot them
def find_matching_sweep(dataset,df, folder,clamp_mode='not_clamped', show_plot=True):
    min_difference = float('inf')
    closest_indexes = None
    series1 = df[(df.condition.values == 'baseline') & (df.clamp_mode.values == clamp_mode)].sag_volt
    series2 = df[(df.condition.values == 'naag') & (df.clamp_mode.values == clamp_mode)].sag_volt
    for i, value1 in enumerate(series1):
        for j, value2 in enumerate(series2):
            difference = abs(value1 - value2)
            if difference < min_difference:
                # Update the closest pair and minimum difference
                min_difference = difference
                closest_indexes = (series1.index[i], series2.index[j])
    if closest_indexes is not None:
        closest_sweeps=(df.loc[closest_indexes[0]].sweep_number,df.loc[closest_indexes[1]].sweep_number)
        closest_sags=(df.loc[closest_indexes[0]].sag_ratio,df.loc[closest_indexes[1]].sag_ratio)
        closest_conds=(df.loc[closest_indexes[0]].condition,df.loc[closest_indexes[1]].condition)
        closest_volts=(df.loc[closest_indexes[0]].sag_volt,df.loc[closest_indexes[1]].sag_volt)
        sweep_numbers = list(closest_sweeps)
        sag_ratios = list(closest_sags)
        sag_voltages = list(closest_volts)
        conditions = list(closest_conds)
        output = pd.DataFrame({'sweep_number': sweep_numbers,'clamp_mode':clamp_mode, 'condition': conditions,'sag_volt': sag_voltages,'sag_ratio': sag_ratios, 'cell':df.loc[closest_indexes[1]].cell})
        output.loc[output.condition.values == 'baseline', 'norm_sagratio'] = 1
        output.loc[output.condition.values == 'naag', 'norm_sagratio'] = output[output.condition.values == 'naag'].sag_ratio.values[0]/output[output.condition.values == 'baseline'].sag_ratio.values[0]
        if show_plot:
            plt.figure()
            a=downsample(dataset,sweep_numbers[0])
            b=downsample(dataset,sweep_numbers[1])
            #plt.plot(dataset.sweep(sweep_number=sweep_numbers[0]).v)
            plt.plot(a)
            plt.plot(b)
            #plt.plot(dataset.sweep(sweep_number=sweep_numbers[1]).v)
            plt.savefig(folder+'/sags1_'+str(sweep_numbers[0])+'_'+str(sweep_numbers[1])+'.eps', format='eps')
    else:
        df = df.reset_index(drop=True)
        output = pd.DataFrame(columns=['sweep_number', 'clamp_mode', 'condition', 'sag_volt', 'sag_ratio', 'cell'])
        output['cell'] = df.cell[0]
    return output
        
#function to find the indexes of array, used in find_peaks
def find_crossing_indexes2(signal, Y):
        crossings = np.where(np.diff(np.sign(signal - Y)))[0]
        if len(crossings) >= 2:
            crossing1 = crossings[0]
            crossing2 = crossings[-1]
            return crossing1, crossing2
        else:
            return None, None
        
#funtion coding for the decay fit of the tau (used in decay tau for in TP analysis and peaks)
def monoExp_cc(x, m, t, b):
    return m * np.exp(-t * x) + b
def monoExp_vc(x, m, t, b):
    return -m * np.exp(-t * x) + b

#finding the peaks on extracellular stimulation recording, 5 and 20 Hz, CC and VC; defaults: CC, baseline and washin conditions
def find_peaks(dataset,trace1,trace2,freq,path,file,folder,sweeps,trace3=None,cmode='CC',note='non-spiking'): #add here traces later !
    peaks_results = pd.DataFrame()
    fs=dataset.sweep(sweep_number=0).sampling_rate
    traces = {'baseline':trace1, 'naag':trace2}
    if len(traces['baseline']) != len(traces['naag']):
        length_baseline=len(traces['baseline'])
        length_naag=len(traces['naag'])
        scaling_factor = math.ceil(max(length_baseline, length_naag) / min(length_baseline, length_naag))
        corr_len=min(length_baseline, length_naag)
        if length_baseline > length_naag:
            traces['baseline']=traces['baseline'][::scaling_factor]
        elif length_naag > length_baseline:
            traces['naag']=traces['naag'][::scaling_factor]
    else:
        corr_len=len(traces['baseline'])
    if trace3 is not None:
        traces['washout'] = trace3
        #add here later a piece of code which does the same for washout
    art=90
    #get the input arguments
    f = h5py.File(path+'/'+file,'r')
    ipts=pd.DataFrame(list(f['stimulus']['presentation'].keys()), columns=['ttl'])
    data_TTL = ipts[ipts['ttl'].str.contains('TTL')]
    for k in traces.keys(): #once I find out how to get the TTL trace, replace with one line of code
        if freq == 5:
            s=str(sweeps[sweeps.stimulus_code.str.contains('baa')].reset_index(drop=True).sweep_number[0])
            swps=data_TTL.loc[data_TTL.ttl.str.contains(s)].reset_index(drop=True)
            ttl = np.array(f['stimulus']['presentation'][swps.ttl[0]]['data'])
            if len(ttl)==corr_len:
                stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
            else:
                factor=math.ceil(len(ttl)/corr_len)
                ttl=ttl[::factor]
                stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
        elif freq == 20:
            s=str(sweeps[sweeps.stimulus_code.str.contains('20')].reset_index(drop=True).sweep_number[0])
            swps=data_TTL.loc[data_TTL.ttl.str.contains(s)].reset_index(drop=True)
            ttl = np.array(f['stimulus']['presentation'][swps.ttl[0]]['data'])
            if len(ttl) ==corr_len:
                stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
            else:    
                factor=math.ceil(len(ttl)/corr_len)
                ttl=ttl[::factor]
                stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
        else:
            raise ValueError("Invalid value for 'freq'. Supported values are 5 and 20 Hz")
        #analyse the trace
        trace = traces[k]
        peaks=[]
        delta = stimind[1]-stimind[0]
        for i in stimind:
            if cmode == 'CC':
                if i != stimind[-1]:
                    ind=np.argmax(trace[i+art:i+delta])
                    peaks.append(ind+i+art)
                else:
                    ind=np.argmax(trace[i+art:i+delta])
                    peaks.append(ind+i+art)
            else:
                if i != stimind[-1]:
                    ind=np.argmin(trace[i+art:i+delta])
                    peaks.append(ind+i+art)
                else:
                    ind=np.argmin(trace[i+art:i+delta])
                    peaks.append(ind+i+art)
        plt.figure()
        plt.plot(trace)
        plt.plot(peaks, trace[peaks], "x")
        plt.title(k)
        peak_results = pd.DataFrame(columns = ['peak_number','baseline_index','peak_index','baseline_voltage','peak_voltage', 
                                                   'amplitude','local_amplitude','delay(ms)','delay_to_peak(ms)','delay(diff)','rise2080(ms)','halfwidth',
                                                   'tau(ms)','ahp','delta_ahp'])
        if len(peaks) == 9:
            for number, location in enumerate(peaks):
                if freq == 5:
                    peak_nr = number+1
                    baseline_index = int(stimind[number] - (delta/10))
                    baseline_value = trace[baseline_index]
                    peak_index=location
                    peak_voltage = trace[location]
                    peak_ampl = peak_voltage - baseline_value
                    #subselection a peak from the trace to calculate risetime, halfwidth
                    signal_start=np.where(np.diff(np.sign(trace[baseline_index-art:location] - baseline_value)))[0][-1]+baseline_index-art
                    if peak_nr != 9:
                        signal = trace[signal_start:stimind[number+1]-50]
                    else:
                        signal = trace[signal_start:int(signal_start+delta)]
                    #finding the delay tot he beginning of the peak (not as accurate) and to the peak (accurate)
                    dv = np.gradient(trace, edge_order =1)
                    t=np.linspace(1,len(trace), num=len(trace))
                    dt = np.gradient(t, edge_order =1)
                    dvdt = dv/dt
                    delay_start=np.argmax(dvdt[stimind[number]+art:stimind[number]+500])+stimind[number]
                    delay=(delay_start-stimind[number])/fs*1000
                    delay_to_peak = (peak_index-delay_start)/fs*1000
                    delay_diff=(signal_start-stimind[number])/fs*1000
                    if cmode == 'CC':
                        peak_id=np.argmax(signal)
                    else:
                        peak_id=np.argmin(signal)
                    #risetime 20-80
                    range_signal = max(signal[:peak_id]) - min(signal[:peak_id])
                    if cmode == 'CC':
                        val20=min(signal[:peak_id])+range_signal*0.2
                        val80=min(signal[:peak_id])+range_signal*0.8
                    else: 
                        val20=max(signal[:peak_id])-range_signal*0.2
                        val80=max(signal[:peak_id])-range_signal*0.8
                    rise20 = np.where(np.diff(np.sign(signal[:peak_id] - val20)))[0][0]
                    rise80 = np.where(np.diff(np.sign(signal[:peak_id] - val80)))[0][0]
                    rise_time = (rise80 - rise20) / fs*1000
                    #halfwidth
                    if cmode == 'CC':
                        half_val=signal[peak_id]-range_signal/2
                    else:
                        half_val=signal[peak_id]+range_signal/2
                    crossing1, crossing2 = find_crossing_indexes2(signal, half_val)
                    halfwidth=(crossing2 - crossing1) / fs *1000
                    #decay fit
                    decay_signal=signal[peak_id:]
                    x = np.arange(0, len(decay_signal))
                    m_guess = peak_ampl
                    b_guess = baseline_value
                    t_guess = 0.03
                    if cmode == 'CC':
                        (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp_cc, x, decay_signal, [m_guess, t_guess, b_guess],maxfev=3000)
                    else:
                        (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp_vc, x, decay_signal, [m_guess, t_guess, b_guess],maxfev=3000)
                    decay_tau=-1/(-t_true)/fs*1000
                    #ahp
                    ahp=np.where(np.diff(np.sign(signal - signal[0])))[0]
                    if len(ahp)==1:
                        ahp_value=min(signal[-1000:])
                    else:
                        ahp_start=np.where(np.diff(np.sign(signal - signal[0])))[0][1]
                        ahp_value=min(signal[ahp_start:])
                    ahp_deflection = signal[0]-ahp_value
                    peak_results= peak_results.append({'peak_number': peak_nr, 'baseline_index':baseline_index,'peak_index': location, 
                                                       'baseline_voltage':baseline_value,'peak_voltage': peak_voltage, 
                                                       'amplitude': peak_ampl,'local_amplitude':peak_ampl,'delay(ms)':delay,'delay_to_peak(ms)':delay_to_peak,'delay(diff)':delay_diff,
                                                       'rise2080(ms)':rise_time,'halfwidth':halfwidth,
                                                       'tau(ms)':decay_tau,'ahp':ahp_value,'delta_ahp':ahp_deflection}, ignore_index=True)
                    peak_results['condition'] = k+'_'+str(freq)
                    peak_results['clamp_mode'] = cmode
                    peak_results['cell'] = file
                elif freq==20:
                    peak_voltage = trace[location]
                    peak_nr = number+1
                    total_baseline=trace[stimind[0] - 500]
                    if number==0 or number==8:
                        baseline_index = int(stimind[number] - 300)
                        baseline_value = trace[baseline_index]
                        peak_ampl = peak_voltage - total_baseline
                        rpeak_ampl = peak_voltage - baseline_value
                        peak_results= peak_results.append({'peak_number': peak_nr, 'baseline_index':baseline_index,'peak_index': location, 
                                                           'baseline_voltage':baseline_value,'peak_voltage': peak_voltage, 
                                                           'amplitude': peak_ampl,'local_amplitude':rpeak_ampl,'delay(ms)':'nan','delay_to_peak(ms)':'nan',
                                                           'rise2080(ms)':'nan','halfwidth':'nan','tau(ms)':'nan','ahp':'nan','delta_ahp':'nan'}, ignore_index=True)
                        peak_results['condition'] = k+'_'+str(freq)
                        peak_results['clamp_mode'] = cmode
                        peak_results['cell'] = file
                    else:
                        baseline_index = int(stimind[number] - 200)
                        baseline_value = trace[baseline_index]
                        peak_ampl = peak_voltage - total_baseline
                        rpeak_ampl=peak_voltage-baseline_value
                        peak_results= peak_results.append({'peak_number': peak_nr, 'baseline_index':baseline_index,'peak_index': location, 
                                                           'baseline_voltage':baseline_value,'peak_voltage': peak_voltage, 
                                                           'amplitude': peak_ampl,'local_amplitude':rpeak_ampl,'delay(ms)':'nan','delay_to_peak(ms)':'nan','delay(diff)':'nan',
                                                           'rise2080(ms)':'nan','halfwidth':'nan','tau(ms)':'nan','ahp':'nan','delta_ahp':'nan'}, ignore_index=True)
                        peak_results['condition'] = k+'_'+str(freq)
                        peak_results['clamp_mode'] = cmode
                        peak_results['cell'] = file
        else:
                peak_results= peak_results.append({'peak_number': 'nan', 'baseline_index':'nan','peak_index': 'nan', 
                                                   'baseline_voltage':'nan','peak_voltage': 'nan', 
                                                   'amplitude': 'nan','local_amplitude': 'nan','delay(ms)':'nan','delay(diff)':'nan',
                                                   'delay_to_peak(ms)':'nan','rise2080(ms)':'nan','halfwidth':'nan',
                                                   'tau(ms)':'nan','ahp':'nan','delta_ahp':'nan'}, ignore_index=True)         
                peak_results['condition'] = k+'_'+str(freq)+'_'+cmode
                peak_results['clamp_mode'] = cmode
                peak_results['cell'] = file
        peaks_results = pd.concat([peaks_results,peak_results], axis=0)
    plt.figure()
    plt.plot(trace1)
    plt.plot(trace2)
    if trace3 is not None:
        plt.plot(trace3)
    plt.title(file)
    plt.savefig(folder+'/'+str(freq)+'_'+cmode+'.eps', format='eps')
    bas=peaks_results[peaks_results.condition.values == 'baseline_'+str(freq)]
    naag=peaks_results[peaks_results.condition.values == 'naag_'+str(freq)]
    naag_ppr=naag[naag['peak_number'].values == 2].amplitude.values[0] / naag[naag['peak_number'].values == 1].amplitude.values[0]
    bas_rec = bas[bas['peak_number'].values == 9].amplitude.values[0] / bas[bas['peak_number'].values == 1].amplitude.values[0]
    naag_rec=naag[naag['peak_number'].values == 9].amplitude.values[0] / naag[naag['peak_number'].values == 1].amplitude.values[0]
    bas_ppr = bas[bas['peak_number'].values == 2].amplitude.values[0] / bas[bas['peak_number'].values == 1].amplitude.values[0]
    peaks_results.loc[peaks_results.condition.values == 'baseline_'+str(freq), 'ppr'] = bas_ppr
    peaks_results.loc[peaks_results.condition.values == 'naag_'+str(freq), 'ppr'] = naag_ppr
    peaks_results.loc[peaks_results.condition.values == 'baseline_'+str(freq), 'recovery'] = bas_rec
    peaks_results.loc[peaks_results.condition.values == 'naag_'+str(freq), 'recovery'] = naag_rec
    if trace3 is not None:
        was = peaks_results[peaks_results.condition.values == 'washout_'+str(freq)]
        was_ppr=was[was['peak_number'].values == 2].amplitude.values[0] / was[was['peak_number'].values == 1].amplitude.values[0]
        was_rec = was[was['peak_number'].values == 9].amplitude.values[0] / was[was['peak_number'].values == 1].amplitude.values[0]
        peaks_results.loc[peaks_results.condition.values == 'washout_'+str(freq), 'ppr'] = was_ppr
        peaks_results.loc[peaks_results.condition.values == 'washout_'+str(freq), 'recovery'] = was_rec
    if note == 'spiking':
        # Perform additional processing and return extra output based on 'spiking' note
        a=traces['naag']
        what=sweeps['stimulus_code'].str.contains('naag') & ~sweeps['stimulus_code'].str.contains('CC|20|Pulse|rmp')
        frstswp=sweeps[what].reset_index(drop=True)['sweep_number'][0]
        b=dataset.sweep(sweep_number=frstswp).i
        c=dataset.sweep(sweep_number=frstswp).t
        ext = SpikeFeatureExtractor()
        extra_output = ext.process(c, a, b)
        return peaks_results, extra_output
    else:
        return peaks_results

#finding the peaks on extracellular stimulation recording, 100Hz, CC and VC, defaults: two conditions
def find_peaks100(trace1,trace2,path,file, folder, sweeps,trace3=None):
    peak_results100=pd.DataFrame()
    traces={'baseline':trace1,'naag':trace2}
    if trace3 is not None:
        traces['washout'] = trace3
    art=100
    f = h5py.File(path+'/'+file,'r')
    ipts=pd.DataFrame(list(f['stimulus']['presentation'].keys()), columns=['ttl'])
    data_TTL = ipts[ipts['ttl'].str.contains('TTL')]
    s=str(sweeps[sweeps.stimulus_code.str.contains('100')].reset_index(drop=True).sweep_number[0])
    swps=data_TTL.loc[data_TTL.ttl.str.contains(s)].reset_index(drop=True)
    ttl = np.array(f['stimulus']['presentation'][swps.ttl[0]]['data'])
    if len(ttl) ==29502 or len(ttl) ==29500:
        stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
    else:
        factor=int(len(ttl)/29502)
        if factor !=0:
            ttl=ttl[::factor]
            stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
    for k in traces.keys():#list of the files for single traces
        trace = traces[k]
        peaks=[]
        for i in stimind:
            if i != stimind[-1]:
                ind=np.argmax(trace[i+art:i+250])
                peaks.append(ind+i+art)
            else:
                ind=np.argmax(trace[i+art:i+2000])
                peaks.append(ind+i+art)
        plt.figure()
        plt.plot(trace)
        plt.title(k)
        plt.plot(peaks, trace[peaks], "x")
        plt.savefig(folder+'/'+str(100)+'.eps', format='eps')
        peak_results = pd.DataFrame(columns = ['peak_number','baseline_index','peak_index','baseline_voltage','peak_voltage', 
                                                   'amplitude','local_amplitude','delay(ms)','delay_to_peak(ms)','rise2080(ms)','halfwidth',
                                                   'tau(ms)','ahp','delta_ahp'])
        for number,location in enumerate(peaks):
            peak_voltage = trace[location]
            peak_nr = number+1
            total_baseline=trace[stimind[0] - 500]
            peak_ampl = peak_voltage - total_baseline
            peak_results= peak_results.append({'peak_number': peak_nr, 'baseline_index':'nan','peak_index': location, 
                                                           'baseline_voltage':total_baseline,'peak_voltage': peak_voltage, 
                                                           'amplitude': peak_ampl,'local_amplitude':'nan','delay(ms)':'nan','delay_to_peak(ms)':'nan',
                                                           'rise2080(ms)':'nan','halfwidth':'nan','tau(ms)':'nan','ahp':'nan','delta_ahp':'nan'}, ignore_index=True)
            peak_results['condition'] = k+'_'+str(100)
            peak_results['clamp_mode'] = 'CC'
            peak_results['cell'] = file
        peak_results['ppr'] = (peak_results.loc[peak_results['peak_number'] == 2, 'amplitude'].values[0]) / (peak_results.loc[peak_results['peak_number'] == 1, 'amplitude'].values[0])
        peak_results['recovery'] = (peak_results.loc[peak_results['peak_number'] == 9, 'amplitude'].values[0]) / (peak_results.loc[peak_results['peak_number'] == 1, 'amplitude'].values[0])
        peak_results100 = pd.concat([peak_results100,peak_results], axis=0)  
    return peak_results100

#%% output processing
#a function to normalise the values of the table to the first peak of the stimulation
def norm_to_first(df, column, group_columns=['condition']):
    new_column_values = []
    
    for index, row in df.iterrows():
        amplitude = row[column]

        if row['peak_number'] == 1:
            value = 1
        elif row['peak_number'] != 1:
            # Create a mask to filter rows based on group_columns
            mask = True
            for group_col in group_columns:
                mask &= (df[group_col] == row[group_col])

            peak1_amplitude = df[mask & (df['peak_number'] == 1)][column].values

            if len(peak1_amplitude) > 0:
                peak1_amplitude = peak1_amplitude[0]
                value = amplitude / peak1_amplitude
            else:
                value = None

        new_column_values.append(value)

    col_name = f"{column}_normto1"
    df[str(col_name)] = new_column_values
    return df


#a function to normalise the values of the table peakwise to baseline values
def norm_to_baseline(df,column):
    new_column_values = []
    baseline_values = {}
    naag_values={}
    if df['condition'].str.contains('washout').any():
        was_values={}
    #get the values from the table
    for index, row in df.iterrows():
        value = row[column] #change the input column
        condition = row['condition']
        peak_number = row['peak_number']
        if 'baseline' in condition:
            baseline_values[peak_number] = value
        elif 'naag' in condition:
           naag_values[peak_number] = value
        elif 'washout' in condition:
           was_values[peak_number] = value
    #write the new values in the table
    for index, row in df.iterrows():
        condition = row['condition']
        peak_number = row['peak_number']
        if 'baseline' in condition:
            new_value = 1
        elif 'naag' in condition:
            baseline_value = baseline_values.get(peak_number, None)
            naag_value = naag_values.get(peak_number, None)
            if baseline_value is not None and naag_value is not None:
                new_value = naag_value / baseline_value
            else:
                new_value = None
        elif 'washout' in condition:
            baseline_value = baseline_values.get(peak_number, None)
            was_value = was_values.get(peak_number, None)
            if baseline_value is not None and was_value is not None:
                new_value = was_value / baseline_value
            else:
                new_value = None
        else:
            new_value = None
        new_column_values.append(new_value)
    col_name = column + '_normtobaseline'
    df[str(col_name)] = new_column_values
    return df

def normalise(df,column):
    new_column_values = []
    for index, row in df.iterrows():
        cell = row['cell']
        condition = row['condition']
        amplitude = row[column] #change the input column
        # Check the condition and calculate the corresponding value
        if condition == 'baseline':
            baseline_amplitude = amplitude
            if amplitude==0:
                value='nan'
            else:
                value = amplitude/amplitude
        elif condition == 'naag' or condition == 'washout':
            baseline_amplitude = df[(df['cell'] == cell) & (df['condition'] == 'baseline')][column].values[0] #change the input column
            value = amplitude / baseline_amplitude
        else:
            value = None
        # Append the calculated value to the list
        new_column_values.append(value)
    new_name='norm_'+column
    df[new_name]=new_column_values
    return df


def get_active(savedir,df, mode='not'):
    output=pd.DataFrame()
    for cell in df.cell.unique():
        df2=df[df.cell.values == cell]
        for cond in df2.condition.unique():
            df3=df2[df2.condition.values == cond].reset_index(drop=True)
            first_row_index = df3[df3['nopaps'] > 0].index[0]
            first_row = df3.loc[first_row_index]
            columns_needed = ['cell', 'condition','current', 'first_upstroke','first_downstroke', 'first_threshold', 'first_ap_v', 'first_width', 'first_updownratio', 'species', 'Group']
            row = first_row[columns_needed]  
            output = output.append(row, ignore_index=True)
    for cell1 in output.cell.unique():
        if len(output.loc[output.cell == cell1]) == 1:
            output = output[output.cell != cell1]
    #output = output.groupby('cell').filter(lambda x: len(x) > 1)
    for i in output.columns[2:-2]:
        output=normalise(output,i)
    if mode == 'not':
        output_filename = savedir[:-8] + str(current_date) + 'actives_not_total.csv'
    elif mode == 'cl':
        output_filename = savedir[:-8] + str(current_date) + 'actives_cl_total.csv'
    output.to_csv(output_filename)
    return output
       
#function(s) for bulk concatination of all the cells into one table
def bulk_concat(directory,list1):
    output = pd.DataFrame()
    for t1 in list1: 
        temp = pd.read_csv(directory + '/' + t1)
        output = pd.concat([output,temp], axis=0, ignore_index = True)
    output.loc[:,'species']=pd.Series([output.cell[i][0] for i in range(0,len(output))])
    return output   
#assigning group to the cell based on metadata
def group_ass(df, data_dict):
    for index, row in df.iterrows():
        reference_value = row['cell'][:-4]
        if reference_value in data_dict:
            info_to_add = data_dict[reference_value]
            # Create the 'Group' column if it doesn't exist
            if 'Group' not in df.columns:
                df['Group'] = None
            # Update the row in the 'Group' column with the retrieved information
            df.at[index, 'Group'] = info_to_add
    return df
#saving the data and loading it all, based on functions above
def process_and_save_data(savedir, filenames_prefix, data_dict):
    files = [t for t in os.listdir(savedir) if t.startswith(filenames_prefix)]
    files = sorted(files)

    total_data = bulk_concat(savedir, files)
    total_data = group_ass(total_data, data_dict)

    output_filename = savedir[:-8] + str(current_date) + f'_{filenames_prefix}_total.csv'
    total_data.to_csv(output_filename)
    return total_data
#%%analysis per set of protocols which are looped
#sweeps processing
def process_sweep_table2(file,df,df2):
    df['cell']=file[:-4]
    ref_df=df2[df2.cell.values == df.cell[0]].reset_index(drop=True)
    
    #write into that the conditions from the metadata table
    cond1 = int(ref_df[ref_df.condition == 'baseline']['frstswp_condition'].iloc[0])
    conditions = [(cond1 <= df.index)]
    values = ['baseline']

    # Check and update for 'wash'
    if 'wash' in ref_df.condition.values:
        cond2 = int(ref_df[ref_df.condition == 'wash']['frstswp_condition'].iloc[0])
        conditions[0] = (cond1 <= df.index) & (df.index < cond2)  # Update the upper border for baseline
        conditions.append((cond2 <= df.index))  # Condition for 'wash'
        values.append('wash')

    # Check and update for 'naag'
    if 'naag' in ref_df.condition.values:
        cond3 = int(ref_df[ref_df.condition == 'naag']['frstswp_condition'].iloc[0])
        if 'wash' in values:
            conditions[-1] = (cond2 <= df.index) & (df.index < cond3)  # Update the upper border for wash
        conditions.append((cond3 <= df.index))  # Full condition for 'naag'
        values.append('naag')

    # Apply conditions to the dataframe
    df['condition'] = np.select(conditions, values, default='unknown')
    #find the beginning of each set
    sets=df['stimulus_code'].str.contains('rmp', case=False)
    #how many sets there are? 
    sets_amount = sets.sum()
    #indexes of the sets starts
    location = sets[sets == True].index
    #locate the set_number at the index in the original df
    df.loc[location, 'set_number'] = range(1, sets_amount + 1)
    #fill in the rest of the set
    df['set_number'] = df['set_number'].fillna(method='ffill')
    df['set_number'][0]=1
    return df


#find peaks in stimulation protocols
def find_peaks_set(dataset,path,file,folder,sweeps_cleared):  
    peaks_results = pd.DataFrame(columns = ['cell','set_number','condition','frequency','peak_number','total_baseline', 
                                                'local_baseline', 'total_amplitude', 'local_amplitude','peak_v', 'halfwidth(ms)','risetime2080(ms)',
                                                'decay_tau(ms)', 'ahp_abs', 'ahp_deflection'])  
    spike_results=pd.DataFrame(columns = ['cell','set_number','condition','frequency','sweep_number',
                                                            'peak_number','threshold_v', 'peak_v','upstroke', 'downstroke','halfwidth','spike_p'])
    if not os.path.exists(folder+'/stim/'):
        os.makedirs(folder+'/stim/')
    for set_number in sweeps_cleared.set_number.unique():
        print(f'-------------set{set_number}------------------')
        df=sweeps_cleared[sweeps_cleared.set_number.values == int(set_number)].reset_index(drop=True)    
        #find the stimulations at rmp
        # df4=df[(df.stimulus_code.str.contains(f'_{df.condition[0]}_', regex=True))&(df.leak_pa.isnull())].reset_index(drop=True)
        df4=df[(df.stimulus_code.str.contains('_stim_', regex=True))&(df.leak_pa.isnull())].reset_index(drop=True)
        if df4.empty:
            print(f"Skipping set {set_number} as df4 is empty.")
            continue
        condition=df4.condition[0]
        #initiate the table with results for peaks
        for stim in df4.stimulus_code.unique():
            df4_sub = df4[df4.stimulus_code == stim]
            art=90
            #get the input arguments
            #f = h5py.File(path+'/'+file,'r')
            f = h5py.File(path+'/all_data/'+file+'.nwb','r')
            ipts=pd.DataFrame(list(f['stimulus']['presentation'].keys()), columns=['ttl'])
            data_TTL = ipts[ipts['ttl'].str.contains('TTL')]
            s=str(sweeps_cleared[sweeps_cleared.stimulus_code.str.contains(stim)].reset_index(drop=True).sweep_number[0])
            fs=dataset.sweep(sweep_number=int(s)).sampling_rate
            swps=data_TTL.loc[data_TTL.ttl.str.contains(s)].reset_index(drop=True)
            ttl = np.array(f['stimulus']['presentation'][swps.ttl[0]]['data'])
            stimind=[i for i in range(1, len(ttl) - 1) if ttl[i] == 2 and ttl[i] - ttl[i-1] == 2 and ttl[i + 1] - ttl[i] == 0]
            #identify the peaks
            delta = stimind[1]-stimind[0]
            freq=fs/delta
            peaks=[]
            if 'spiking' not in df4_sub.status.values:
                avg=get_avg(dataset,sweeps_cleared,stim,'CurrentClamp',df4.set_number[0], mode=0)
                for i in stimind:
                    ind=np.argmax(avg[i+art:i+delta])
                    peaks.append(ind+i+art)
                # Plot the average with peaks
                plt.figure()
                plt.plot(avg)
                plt.plot(peaks, avg[peaks], "x")
                plt.title(str(int(set_number))+'_'+str(stim)+'_'+file)
                plt.savefig(folder+'/stim/'+str(int(set_number))+'_'+str(stim)+'_'+file[:-4]+'.eps', format='eps') 
                #get parameters for the peaks and plot them
                fig, axes = plt.subplots(1, 6, figsize=(15, 10))
                fig.suptitle(f'{int(set_number)}_{stim}_{file} - Peaks Analysis', fontsize=16)
                if len(peaks) == 6:
                    total_baseline=avg[int(stimind[0]-(delta/10))]
                    for number, location in enumerate(peaks):
                        peak_nr = number+1
                        baseline_index = int(stimind[number] - (delta/10))
                        baseline_value = avg[baseline_index]
                        peak_voltage = avg[location]
                        peak_ampl = peak_voltage - baseline_value
                        syn_bup_ampl=peak_voltage-total_baseline
                        #subselection a peak from the trace to calculate risetime, halfwidth
                        signal_start=np.where(np.diff(np.sign(avg[baseline_index-art:location] - baseline_value)))[0][-1]+baseline_index-art
                        if peak_nr in range(1,5):
                            signal = avg[signal_start:stimind[number+1]-50]
                        else:
                            signal = avg[signal_start:int(signal_start+delta)]
                        #risetime 20-80
                        peak_id=np.argmax(signal)
                        range_signal = max(signal[:peak_id]) - min(signal[:peak_id])
                        val20=min(signal[:peak_id])+range_signal*0.2
                        val80=min(signal[:peak_id])+range_signal*0.8
                        rise20 = np.where(np.diff(np.sign(signal[:peak_id] - val20)))[0][0]
                        rise80 = np.where(np.diff(np.sign(signal[:peak_id] - val80)))[0][0]
                        risetime = (rise80 - rise20) / fs*1000
                        #halfwidth
                        half_val=signal[peak_id]-range_signal/2
                        crossing1, crossing2 = find_crossing_indexes2(signal, half_val)
                        halfwidth=(crossing2 - crossing1) / fs *1000
                        #decay fit
                        decay_signal=signal[peak_id:]
                        x = np.arange(0, len(decay_signal))
                        m_guess = peak_ampl
                        b_guess = baseline_value
                        t_guess = 0.03
                        (m_true, t_true, b_true), cv = sp.optimize.curve_fit(monoExp_cc, x, decay_signal, [m_guess, t_guess, b_guess],maxfev=10000)
                        decay_tau=-1/(-t_true)/fs*1000
                        #ahp
                        ahp=np.where(np.diff(np.sign(signal - signal[0])))[0]
                        if len(ahp)==1:
                            ahp_value=min(signal[-1000:])
                        else:
                            ahp_start=np.where(np.diff(np.sign(signal - signal[0])))[0][1]
                            ahp_value=min(signal[ahp_start:])
                        ahp_deflection = signal[0]-ahp_value
                        #plotting 
                        axes[number].plot(signal)
                        axes[number].plot(peak_id,signal[peak_id],'x', label = 'Peak')
                        axes[number].axhline(half_val, color='r', label = 'halfwidth')
                        axes[number].axhline(val80, color='c', linestyle = '--', label = '80%')
                        axes[number].axhline(val20, color='k', linestyle = '--',label = '20%')
                        x_fit = np.arange(0, len(x)) + peak_id
                        axes[number].plot(x_fit, monoExp_cc(x, m_true, t_true, b_true), '--', label='Decay Fit')
                        axes[number].legend()
                        axes[number].set_title(f'peak{peak_nr}')
    
                        peaks_results= peaks_results.append({'cell':file,'set_number':int(set_number),'condition':condition,
                                                             'frequency':freq,'peak_number':peak_nr,'total_baseline':total_baseline, 
                                                            'local_baseline':baseline_value, 'total_amplitude':syn_bup_ampl,
                                                            'local_amplitude':peak_ampl,'peak_v':peak_voltage, 'halfwidth(ms)':halfwidth,
                                                            'risetime2080(ms)':risetime,'decay_tau(ms)':decay_tau,
                                                            'ahp_abs':ahp_value, 'ahp_deflection':ahp_deflection}, ignore_index=True)
            
                plt.tight_layout()  # Adjust layout for better spacing
                min_y = float('inf')
                max_y = float('-inf')
                
                # Find overall min and max values among all subplots
                for ax in axes:
                    ymin, ymax = ax.get_ylim()
                    min_y = min(min_y, ymin)
                    max_y = max(max_y, ymax)
                
                # Set the same Y-axis limits for all subplots
                for ax in axes:
                    ax.set_ylim(min_y, max_y)   
                plt.savefig(folder+'/stim/'+str(int(set_number))+'_'+str(stim)+'_'+file[:-4]+'_peaks.eps', format='eps') 
            else:
                peaks_results = peaks_results.append({'cell': file, 'set_number': int(set_number), 'condition': condition,
                                              'frequency': freq, 'peak_number': 'spiking', 'total_baseline': 'spiking', 
                                              'local_baseline': 'spiking', 'total_amplitude': 'spiking',
                                              'local_amplitude': 'spiking','peak_v':'spiking', 'halfwidth(ms)': 'spiking',
                                              'risetime2080(ms)': 'spiking', 'decay_tau(ms)': 'spiking',
                                              'ahp_abs': 'spiking', 'ahp_deflection': 'spiking'}, ignore_index=True)
                for index, row in df4_sub.iterrows():
                    if row['status'] == 'spiking':
                        # get the spike data
                        swp_nr=row['sweep_number']
                        sweep=dataset.sweep(sweep_number=swp_nr)
                        ext = SpikeFeatureExtractor()
                        results = ext.process(sweep.t, sweep.v, sweep.i)
                        plt.figure()
                        plt.plot(sweep.v)
                        plt.plot(results.peak_index, sweep.v[results.peak_index], "x")
                        plt.title(str(int(set_number))+'_'+str(stim)+'_'+file)
                        plt.savefig(folder+'/stim/'+str(int(set_number))+'_'+str(stim)+'_'+file[:-4]+'.eps', format='eps') 
                        # looping over the results table to get the parameters saved into the output table
                        for index2, row2 in results.iterrows():
                            peak_nr=stimind.index(min(stimind, key=lambda x: abs(x - row2['threshold_index'])))+1
                            spike_results=spike_results.append({'cell':file,'set_number':int(set_number),'condition':condition,
                                                            'frequency':freq,'sweep_number':swp_nr,
                                                            'peak_number':peak_nr,'threshold_v':row2['threshold_v'], 'peak_v':row2['peak_v'],
                                                            'upstroke':row2['upstroke'], 'downstroke':row2['downstroke'],
                                                            'halfwidth':row2['width'],'spike_p':len(results)/len(stimind)}, ignore_index=True)
       
        print(f'-------------end of set{set_number}------------------')
    peaks_results=norm_to_first(peaks_results, 'total_amplitude', group_columns=['frequency', 'set_number'])
    peaks_results=norm_to_first(peaks_results, 'local_amplitude', group_columns=['frequency', 'set_number'])
    peaks_results.rename(columns={'total_amplitude_normto1': 'synaptic_integration','local_amplitude_normto1': 'peak_ratios'}, inplace=True)
    peaks_results.to_csv(folder+'/stim_'+file[:-4]+'.csv') 
    return peaks_results, spike_results



def passive_sag_set(dataset,sweeps_cleared,file, folder, mode=1):
    output=pd.DataFrame(columns = ['cell', 'set_number', 'condition','clamp_mode', 'rmp', 'rin_fit', 'sag_fit'])
    sags_total=pd.DataFrame(columns = ['cell', 'set_number', 'sweep_number','curr_inj', 'condition','clamp_mode', 'b_voltage', 'ss_volt', 'sag_volt','sag_ratio','sag_ampl', 'volt_defl', 'sag?'])
    if not os.path.exists(folder+'/sags/'):
        os.makedirs(folder+'/sags/')
    for set_number in sweeps_cleared.set_number.unique():
        df=sweeps_cleared[sweeps_cleared.set_number.values == int(set_number)].reset_index(drop=True)
        #find rmp from the rmp protocol
        rmp_swpnr=df.loc[df.stimulus_code.str.contains('rmp')].reset_index(drop=True)['sweep_number'][0]
        rmp_swp=dataset.sweep(sweep_number=rmp_swpnr).v
        rmp=np.mean(rmp_swp[dataset.sweep(sweep_number=rmp_swpnr).epochs['test'][1]:])
        #find the sag and rin from hypopolarasing steps at rmp
        df2=df[(df.stimulus_code.str.contains('hypo'))&(df.leak_pa.isnull())]
        if df2.shape[0] != 0:
            injs=[]
            rins=[]
            sags=[]
            for index, row in df2.iterrows():
                #select the stimulation epoch of your sweep (epoch 'stim')
                sweep = dataset.sweep(sweep_number=row.sweep_number)
                sweep_number=sweep.sweep_number
                if sweep.epochs['stim'] is not None:
                    start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                    dur=int(abs(start_idx-end_idx))
                    input_trace = sweep.i[:end_idx]
                    if mode == 1:
                        output_trace= artefact_filtering(dataset,sweep.v[:end_idx], show_plot=False)
                    else:
                        output_trace = sweep.v[:end_idx]
                curr_inj = float(round(min(input_trace[start_idx:end_idx])))
                if curr_inj <0:
                    sag_volt_ind = np.where(output_trace==min(output_trace[start_idx:round(start_idx+(dur/5))]))[0][0]
                    sag_volt = np.mean(output_trace[sag_volt_ind-25:sag_volt_ind], axis=0)
                    ss_volt = np.mean(output_trace[round(end_idx-(dur/5)):end_idx], axis=0)
                    base = np.mean(output_trace[start_idx-800:start_idx-300], axis=0)
                    mini_delta=ss_volt-sag_volt
                    sag_delta=base-sag_volt
                    sag_ratio = mini_delta/sag_delta
                    if sag_ratio >0:   
                        rin=(ss_volt-base)/curr_inj*1000
                        injs.append(curr_inj)
                        rins.append(rin)
                        sags.append(sag_ratio)
                        plt.figure()
                        plt.plot(output_trace)
                        plt.axhline(sag_volt, color='r', label='sag_voltage')
                        plt.axhline(ss_volt, color='c', label='steady_state_voltage')
                        plt.axhline(base, color='b', label='baseline_voltage')
                        plt.legend()
                        plt.title("Sweep number: {}, curr_inj: {}".format(sweep_number,curr_inj))
                        plt.savefig(folder+'/sags/'+'sag_set'+str(int(set_number))+'_swp'+str(sweep_number)+'_'+file[:-4]+'.eps', format='eps') 
                        sags_total=sags_total.append({'cell':file, 'set_number':set_number, 'sweep_number':sweep_number,'curr_inj':curr_inj, 'condition':row.condition,
                                          'clamp_mode':'not_clamped','b_voltage':base, 'ss_volt':ss_volt, 'sag_volt':sag_volt, 
                                          'sag_ratio':sag_ratio,'sag_ampl':mini_delta,'volt_defl':sag_delta, 'sag?':'yes'}, ignore_index=True)
                    else:
                        sags_total=sags_total.append({'cell':file, 'set_number':set_number, 'sweep_number':sweep_number,'curr_inj':curr_inj, 'condition':row.condition,
                                          'clamp_mode':'not_clamped','b_voltage':base, 'ss_volt':ss_volt, 'sag_volt':sag_volt, 
                                          'sag_ratio':sag_ratio,'sag_ampl':mini_delta,'volt_defl':sag_delta, 'sag?':'no'}, ignore_index=True)
            #filter the sags - omit the negative sags = no sags
            idx_to_drop = [i for i, value in enumerate(sags) if value is None]
            x_clean = [element for index, element in enumerate(injs) if index not in idx_to_drop]
            y_clean = [element for index, element in enumerate(sags) if index not in idx_to_drop]
            #fit the Rin
            if len(injs)&len(rins) !=0:
                fit_coeffs=np.polyfit(injs,rins, 1)
                rin_fit=fit_coeffs[1]
                plt.figure()
                plt.scatter(injs,rins)
                plt.xlabel('Current injection (pA)')
                plt.ylabel('Input resistance (mOhms)')
                plt.title(file)
                fit_line = np.poly1d(fit_coeffs)
                x_fit = np.linspace(min(injs), max(injs), len(injs))  # Generate points for the fit line
                y_fit = fit_line(x_fit)
                plt.plot(x_fit, y_fit, label=f'Fit: {fit_coeffs[0]:.2f}x + {fit_coeffs[1]:.2f}', color='red')
                plt.savefig(folder+'/sags/'+'rin_fit_set'+str(int(set_number))+'_'+file[:-4]+'.eps', format='eps') 
            else:
                continue
            #fit the sags
            if len(x_clean)&len(y_clean) !=0:
                plt.figure()
                fit_coeffs2=np.polyfit(x_clean,y_clean, 1)
                sag_fit=fit_coeffs2[1]
                plt.scatter(x_clean,y_clean)
                plt.xlabel('Current injection (pA)')
                plt.ylabel('Sag ratio')
                plt.title(file)
                fit_line2 = np.poly1d(fit_coeffs2)
                x_fit2 = np.linspace(min(x_clean), max(x_clean), len(x_clean))  # Generate points for the fit line
                y_fit2 = fit_line2(x_fit2)
                plt.plot(x_fit2, y_fit2, label=f'Fit: {fit_coeffs[0]:.2f}x + {fit_coeffs[1]:.2f}', color='red')
                plt.savefig(folder+'/sags/'+'sag_fit_set'+str(int(set_number))+'_'+file[:-4]+'.eps', format='eps') 
            else:
                continue
            #to edit still, collect the results
            output= output.append({'cell':file, 'set_number':set_number, 'condition':df.condition[0],'clamp_mode':'not_clamped', 'rmp':rmp,
                                    'rin_fit':rin_fit, 'sag_fit':sag_fit}, ignore_index=True)
    #z-score sag amplitude across the sweeps and pull the data
    sags_total['sag_ampl_z'] = sags_total.groupby('curr_inj')['sag_ampl'].transform(zscore)  
    #plot the boxplot of pulled sag amplitude data
    plt.figure()
    sns.boxplot(data=sags_total, x='condition', y='sag_ampl_z')
    sns.swarmplot(data=sags_total, x='condition', y='sag_ampl_z', color='k')
    plt.savefig(folder+'/'+'sagsZ_box'+file[:-4]+'.eps', format='eps') 
    #plot the boxplot of rmp data
    plt.figure()
    sns.boxplot(data=output, x='condition', y='rmp')
    sns.swarmplot(data=output, x='condition', y='rmp', color='k')
    plt.savefig(folder+'/'+'rmp_box_'+file[:-4]+'.eps', format='eps') 
    output.to_csv(folder+'/passives_'+file[:-4]+'.csv')
    sags_total.to_csv(folder+'/all_sags_'+file[:-4]+'.csv')  
    return output, sags_total

def passive_sag_set2(dataset,sweeps_cleared,file, folder, mode=1):
    output=pd.DataFrame(columns = ['cell', 'set_number', 'condition','clamp_mode', 'rmp', 'rin_fit', 'sag_fit'])
    sags_total=pd.DataFrame(columns = ['cell', 'set_number', 'sweep_number','curr_inj', 'condition','clamp_mode', 'b_voltage', 'ss_volt', 'sag_volt','sag_ratio','sag_ampl', 'volt_defl', 'sag?'])
    # if not os.path.exists(folder+'/sags/'):
    #     os.makedirs(folder+'/sags/')
    for set_number in sweeps_cleared.set_number.unique():
        df=sweeps_cleared[sweeps_cleared.set_number.values == int(set_number)].reset_index(drop=True)
        #find rmp from the rmp protocol
        rmp_swpnr=df.loc[df.stimulus_code.str.contains('rmp')].reset_index(drop=True)['sweep_number'][0]
        rmp_swp=dataset.sweep(sweep_number=rmp_swpnr).v
        rmp=np.mean(rmp_swp[dataset.sweep(sweep_number=rmp_swpnr).epochs['test'][1]:])
        #find the sag and rin from hypopolarasing steps at rmp
        df2=df[(df.stimulus_code.str.contains('hypo'))&(df.leak_pa.isnull())]
        if df2.shape[0] != 0:
            injs=[]
            rins=[]
            for index, row in df2.iterrows():
                #select the stimulation epoch of your sweep (epoch 'stim')
                sweep = dataset.sweep(sweep_number=row.sweep_number)
                sweep_number=sweep.sweep_number
                if sweep.epochs['stim'] is not None:
                    start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                    dur=int(abs(start_idx-end_idx))
                    input_trace = sweep.i[:end_idx]
                    if mode == 1:
                        output_trace= artefact_filtering(dataset,sweep.v[:end_idx], show_plot=False)
                    else:
                        output_trace = sweep.v[:end_idx]
                curr_inj = float(round(min(input_trace[start_idx:end_idx])))
                if curr_inj <0:
                    sag_volt_ind = np.where(output_trace==min(output_trace[start_idx:round(start_idx+(dur/5))]))[0][0]
                    sag_volt = np.mean(output_trace[sag_volt_ind-25:sag_volt_ind], axis=0)
                    ss_volt = np.mean(output_trace[round(end_idx-(dur/5)):end_idx], axis=0)
                    base = np.mean(output_trace[start_idx-800:start_idx-300], axis=0)
                    mini_delta=ss_volt-sag_volt
                    sag_delta=base-sag_volt
                    sag_ratio = mini_delta/sag_delta
                    rin=(ss_volt-base)/curr_inj*1000
                    injs.append(curr_inj)
                    rins.append(rin)
                    if sag_ratio >0 and row['status'] == 'have sag':   
                        plt.figure()
                        plt.plot(output_trace)
                        plt.axhline(sag_volt, color='r', label='sag_voltage')
                        plt.axhline(ss_volt, color='c', label='steady_state_voltage')
                        plt.axhline(base, color='b', label='baseline_voltage')
                        plt.legend()
                        plt.title("Sweep number: {}, curr_inj: {}".format(sweep_number,curr_inj))
                        plt.savefig(folder+'/sags/'+'sag_set'+str(int(set_number))+'_swp'+str(sweep_number)+'_'+file[:-4]+'.eps', format='eps') 
                        sags_total=sags_total.append({'cell':file, 'set_number':set_number, 'sweep_number':sweep_number,'curr_inj':curr_inj, 'condition':row.condition,
                                          'clamp_mode':'not_clamped','b_voltage':base, 'ss_volt':ss_volt, 'sag_volt':sag_volt, 
                                          'sag_ratio':sag_ratio,'sag_ampl':mini_delta,'volt_defl':sag_delta, 'sag?':'yes'}, ignore_index=True)
                    else:
                        continue
            
            #fit the Rin
            if len(injs)&len(rins) !=0:
                fit_coeffs=np.polyfit(injs,rins, 1)
                rin_fit=fit_coeffs[1]
                plt.figure()
                plt.scatter(injs,rins)
                plt.xlabel('Current injection (pA)')
                plt.ylabel('Input resistance (mOhms)')
                plt.title(file)
                fit_line = np.poly1d(fit_coeffs)
                x_fit = np.linspace(min(injs), max(injs), len(injs))  # Generate points for the fit line
                y_fit = fit_line(x_fit)
                plt.plot(x_fit, y_fit, label=f'Fit: {fit_coeffs[0]:.2f}x + {fit_coeffs[1]:.2f}', color='red')
                plt.savefig(folder+'/sags/'+'rin_fit_set'+str(int(set_number))+'_'+file[:-4]+'.eps', format='eps') 
            #fit the sags
            sags=list(sags_total[sags_total.set_number == set_number].sag_ratio)
            injs2=list(sags_total[sags_total.set_number == set_number].curr_inj)
            sag_fit='N/A'
            if len(injs2)&len(sags)!=0:
                plt.figure()
                fit_coeffs2=np.polyfit(injs2,sags, 1)
                sag_fit=fit_coeffs2[1]
                plt.scatter(injs2,sags)
                plt.xlabel('Current injection (pA)')
                plt.ylabel('Sag ratio')
                plt.title(file)
                fit_line2 = np.poly1d(fit_coeffs2)
                x_fit2 = np.linspace(min(injs2), max(injs2), len(injs2))  # Generate points for the fit line
                y_fit2 = fit_line2(x_fit2)
                plt.plot(x_fit2, y_fit2, label=f'Fit: {fit_coeffs[0]:.2f}x + {fit_coeffs[1]:.2f}', color='red')
                plt.savefig(folder+'/sags/'+'sag_fit_set'+str(int(set_number))+'_'+file[:-4]+'.eps', format='eps') 
            #to edit still, collect the results
            output= output.append({'cell':file, 'set_number':set_number, 'condition':df.condition[0],'clamp_mode':'not_clamped', 'rmp':rmp,
                                    'rin_fit':rin_fit, 'sag_fit':sag_fit}, ignore_index=True)
    #z-score sag amplitude across the sweeps and pull the data
    sags_total['sag_ampl_z'] = sags_total.groupby('curr_inj')['sag_ampl'].transform(zscore)  
    #plot the boxplot of pulled sag amplitude data
    plt.figure()
    sns.boxplot(data=sags_total, x='condition', y='sag_ampl_z')
    sns.swarmplot(data=sags_total, x='condition', y='sag_ampl_z', color='k')
    plt.savefig(folder+'/'+'sagsZ_box'+file[:-4]+'.eps', format='eps') 
    #plot the boxplot of rmp data
    plt.figure()
    sns.boxplot(data=output, x='condition', y='rmp')
    sns.swarmplot(data=output, x='condition', y='rmp', color='k')
    plt.savefig(folder+'/'+'rmp_box_'+file[:-4]+'.eps', format='eps') 
    output.to_csv(folder+'/passives_'+file[:-4]+'.csv')
    sags_total.to_csv(folder+'/all_sags_'+file[:-4]+'.csv')  
    return output, sags_total





def excitability_set(dataset, sweeps_cleared,folder, file):
    if not os.path.exists(folder+'/IO_curve/'):
        os.makedirs(folder+'/IO_curve/')
    output2=pd.DataFrame()
    output=pd.DataFrame()
    graph = pd.DataFrame(columns=['cell','condition','set_number','current', 'nrofaps', 'first_upstroke','first_downstroke','first_threshold','first_ap_v','first_width','first_updownratio','sweep_nr', 'clamp_mode'])
    for set_number in sweeps_cleared.set_number.unique():
        print(f'-------------set{set_number}------------------')
        df=sweeps_cleared[sweeps_cleared.set_number.values == int(set_number)].reset_index(drop=True)    
        df3=df[df.stimulus_code.str.contains('CCst')]
        if df3.shape[0] != 0:
            for index2, row2 in df3.iterrows():
                sweep = dataset.sweep(sweep_number=row2.sweep_number)
                if sweep.epochs['stim'] is not None:
                    clamp_mode = 'not_clamped' if pd.isnull(row2.leak_pa) else 'clamped'
                    start_idx, end_idx = sweep.epochs['stim'][0], sweep.epochs['stim'][1]
                    ext = SpikeFeatureExtractor()
                    results = ext.process(sweep.t, sweep.v, sweep.i)
                    if len(results)>0:
                        cond=(results['peak_index'] >= start_idx) & (results['peak_index'] <= end_idx)
                        results['sweep_number'] = row2.sweep_number
                        results['condition']=row2.condition
                        results['set_number'] = set_number
                        results['cell']=file
                        results['clamp_mode'] = clamp_mode
                        output2 = pd.concat([output2, results], axis = 0)
                        graph=graph.append({'cell':file,'condition':row2.condition,'set_number':set_number,'current':round(max(sweep.i[start_idx:end_idx],key=abs)),
                                            'nrofaps':len(results[cond]), 'first_upstroke':results['upstroke'].iloc[0],'first_downstroke':results['downstroke'].iloc[0],
                                            'first_threshold':results['threshold_v'].iloc[0],'first_ap_v':results['peak_v'].iloc[0],'first_width':results['width'].iloc[0],
                                            'first_updownratio':results['upstroke_downstroke_ratio'].iloc[0], 'sweep_nr':results['sweep_number'].iloc[0], 'clamp_mode': clamp_mode}, ignore_index=True)
    clamped = graph[graph.clamp_mode.values == 'clamped']
    nclamped = graph[graph.clamp_mode.values == 'not_clamped']
    mapping_dict = dict(zip(nclamped['set_number'], nclamped['condition']))
    output['current']=list(nclamped.groupby(['set_number', 'clamp_mode'])['current'].first())
    output['first_threshold']=list(nclamped.groupby(['set_number', 'clamp_mode'])['first_threshold'].first())
    output['first_ap_v']=list(nclamped.groupby(['set_number', 'clamp_mode'])['first_ap_v'].first())
    output['sweep_nr']=list(nclamped.groupby(['set_number', 'clamp_mode'])['sweep_nr'].first())
    output['set_number']=list(nclamped.set_number.unique())
    output['clamp_mode']='not_clamped'
    output = pd.concat([output,clamped.groupby(['set_number', 'clamp_mode'])[['current', 'first_threshold',
                                                                              'first_ap_v', 'sweep_nr']].first().reset_index()],ignore_index=True)
    output['condition'] =  output['set_number'].map(mapping_dict)
    output['cell'] = file
    #get the slopes into the output
    slopes = []
    for clamp_mode in ['not_clamped', 'clamped']:
        for set_number in graph[graph['clamp_mode'] == clamp_mode]['set_number'].unique():
            # Filter data for the current set number and clamp_mode
            set_data = graph[(graph['set_number'] == set_number) & (graph['clamp_mode'] == clamp_mode)]
            # Perform linear regression (fit a first-degree polynomial)
            coeffs = np.polyfit(list(set_data['current']), list(set_data['nrofaps']), 1)
            # The slope is the first coefficient
            slope = coeffs[0]
            # Append the slope and other information to the list
            slopes.append({'set_number': set_number, 'clamp_mode': clamp_mode, 'slope_current_vs_nrofaps': slope})
    
    # Convert the list of dictionaries to a DataFrame
    slope_df = pd.DataFrame(slopes)
    # Merge the 'output' DataFrame with the 'slope_df' DataFrame on 'set_number' and 'clamp_mode'
    output = pd.merge(output, slope_df, on=['set_number', 'clamp_mode'], how='left')
    
    for index, row in output.iterrows():
        sweep = dataset.sweep(sweep_number=row.sweep_nr)
        ext = SpikeFeatureExtractor()
        results = ext.process(sweep.t, sweep.v, sweep.i)
        plt.figure()
        plt.plot(sweep.v)
        plt.plot(results.peak_index, sweep.v[results.peak_index],'x')
        plt.title(f'{file}, rheobase sweep, set {row.set_number}, {row.clamp_mode}')
        plt.savefig(folder+'/IO_curve/rheobase sweep_set_'+str(int(row.set_number))+'_'+file[:-4]+'.eps', format='eps') 

    output.to_csv(folder+'/actives_'+file[:-4]+'.csv') 
    output2.to_csv(folder+'/total_spikes_'+file[:-4]+'.csv') 
    graph.to_csv(folder+'/total_graph_'+file[:-4]+'.csv') 
    return graph,output,output2


def summarise(passives, actives, peaks_total, spiking_peaks):
    #add up active properties to the passive properties
    test= pd.merge(passives, actives, on=['cell','set_number','condition', 'clamp_mode'], how='inner')
    test.rename(columns={'current': 'rheobase', 'first_threshold': 'rheobase_threshold', 'first_ap_v': 'rheobase_ap_v'}, inplace=True)
    test=test.drop(columns = ['sweep_nr', 'slope_current_vs_nrofaps'])
    to_add=pd.DataFrame(columns=['cell', 'set_number', 'condition', 'stim_frequency', 
                  'max_amplitude','max_v', 'peak1_ampl', 'ppr', 'recovery', 'synaptic_integration','avgfrst_stimspike_th','avgfrst_stimspike_v'])
    for sets in peaks_total.set_number.unique():
        df=peaks_total[peaks_total.set_number==sets].reset_index(drop=True)
        if df.peak_number[0] != 'spiking':
            for freq in df.frequency.unique():
                df2=df[df.frequency==freq].reset_index(drop=True)
                max_ampl=max(df2.total_amplitude)
                max_v=max(df2.peak_v)
                peak1=df2.total_amplitude[0]
                ppr=df2.peak_ratios[1]
                rec=df2.peak_ratios[5]
                syn=df2.synaptic_integration[4]
                to_add = to_add.append({'cell': df2.cell.iloc[0],'set_number': sets,'condition': df2.condition.iloc[0],
                'stim_frequency': freq,'max_amplitude': max_ampl,'max_v':max_v,'peak1_ampl': peak1,'ppr': ppr,'recovery': rec,'synaptic_integration': syn,'avgfrst_stimspike_th':None,'avgfrst_stimspike_v':None}, ignore_index=True)
        else:
            df3=spiking_peaks[spiking_peaks.set_number==df.set_number[0]].reset_index(drop=True)
            for freq in df3.frequency.unique():
                df4=df3[df3.frequency==freq].reset_index(drop=True)
                set_number=df4.set_number[0]
                mean_th=np.mean(df4[df4.peak_number==1].threshold_v)
                mean_ap_v=np.mean(df4[df4.peak_number==1].peak_v)
                to_add = to_add.append({'cell':df4.cell.iloc[0], 'set_number':set_number, 'condition':df4.condition.iloc[0],
                                        'stim_frequency':df4.frequency.iloc[0], 'max_amplitude': None,'max_v':None,'peak1_ampl': None,'ppr': None,'recovery': None,'synaptic_integration': None,
                                        'avgfrst_stimspike_th':mean_th,'avgfrst_stimspike_v':mean_ap_v}, ignore_index=True)
            
    test2= pd.merge(test, to_add, on=['cell','set_number','condition'], how='outer')
    return test2

#%% functions to analyse and plot sEPSPs

# sns.lineplot(data=graph, x='current', y='nrofaps', hue='set_number')
# sns.lineplot(data=graph, x='current', y='nrofaps', hue='condition')
# sns.lineplot(data=graph, x='current', y='nrofaps', hue='set_number')
# plt.figure()
# sns.lineplot(data=graph, x='current', y='nrofaps', hue='condition')
# plt.figure()
# sns.boxplot(data=graph, x='condition', y='first_threshold')
# sns.swarmplot(data=graph, x='condition', y='first_threshold')
# sns.boxplot(data=actives, x='condition', y='first_threshold')
# sns.boxplot(data=actives, x='condition', y='first_threshold')
# nclamped = graph[graph.clamp_mode.values == 'not_clamped']  


def plotting():
    return

def stats(df, param):
    base=df.loc[(df.condition=='baseline')&(~df[param].isna()), param]
    wash=df.loc[(df.condition=='wash')&(~df[param].isna()), param]
    naag=df.loc[(df.condition=='naag')&(~df[param].isna()), param]
    #n_info= 'n_human=' + str(len(hdata)) + ' n_mouse=' + str(len(mdata))
    if (len(hdata)>7) & (len(mdata)>7):
        normal = (stats.normaltest(base)[1]>0.05) & (stats.normaltest(wash)[1]>0.05) & (stats.normaltest(naag)[1]>0.05)
    else:
        normal=False
        
    return
