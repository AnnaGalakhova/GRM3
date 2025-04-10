#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:17:31 2025

@author: annagalakhova
"""

import toolkit
from toolkit import *
from datetime import datetime, timedelta
import pandas as pd
import re
import numpy as np
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

cells_info=pd.read_excel('/Users/annagalakhova/Library/Mobile Documents/com~apple~CloudDocs/PhD INF CNCR VU/DATA/GRM3 project/final_data_assesment/overview_data_final.xlsx')
cells_info=cells_info[~cells_info.cell.isna()]
if '\ufeff' in cells_info['cell'].iloc[0]:
    cells_info['cell'] = cells_info['cell'].str.replace('\ufeff', '')

#define if it is a control or a exprimental cell
cells_info['exp_type'] = None
for idx, row in cells_info.iterrows():
    if pd.isna(row['conditions']):
        continue
    # Get conditions for this row
    conditions = [c.strip() for c in row['conditions'].split(',')]
    
    # Determine experiment type for this row
    if 'vehicle' in conditions:
        cells_info.at[idx, 'exp_type'] = 'control'
    elif 'naag' in conditions:
        cells_info.at[idx, 'exp_type'] = 'experiment'
    else:
        # Default case if neither condition is met
        cells_info.at[idx, 'exp_type'] = 'unknown'
    
del conditions, idx, row
#%% analyse plotted features
rin_cells=cells_info[cells_info.mGluR3_steps_clamped == 1]
rin_total=pd.DataFrame()

for cell in rin_cells.cell:
    ifolder=f'{savepath}/{cell}'
    cell_sweeps=pd.read_csv(f'{ifolder}/{cell}_sweeps.csv', index_col=0)
    group=cells_info[cells_info.cell == cell]['group'].iloc[0]
    species='Mouse' if group == "Mouse" else 'Human'
    exp_type=cells_info[cells_info.cell == cell]['exp_type'].iloc[0]
    #proces the rin table - add the times referenced to the first row of the sweeps table
    cell_rin=pd.read_csv(f'{ifolder}/{cell}_fitted_features.csv' , index_col=0)
    if cell_rin.index.name == 'cell':
       cell_rin=cell_rin.reset_index() 
    set_times=pd.DataFrame(cell_sweeps.groupby(['condition','set_number']).first())[['time_unix','QC','QC_extended']].reset_index() #get times of the sets, starting from first sweep in the set for consistency
    cell_rin = cell_rin.merge(set_times, on=['condition', 'set_number'], how='left') #add time_unix_diff to the dataframe
    #add group and exp_type 
    cell_rin['group']=group
    cell_rin['species']=species
    cell_rin['exp_type']=exp_type
    del set_times
    #add first row for the reference
    frstswp=cell_sweeps.iloc[0]
    new_row = pd.DataFrame({ 'cell':[frstswp.cell], 'clamp_mode':[np.nan], 'set_number':[np.nan], 'condition':[np.nan],
       'condition_set':[np.nan], 'Rin':[np.nan], 'ss_slope':[np.nan], 'sag_slope':[np.nan], 'ratio_of_slopes':[np.nan],
       'rheobase':[np.nan], 'fi_slope':[np.nan], 'time_unix':[frstswp.time_unix]})
    cell_rin = pd.concat([new_row, cell_rin], ignore_index=True) # add the first row as an indicator of the first time point (first recorded sweep)
    del new_row, frstswp
    #add the time
    cell_rin['time_diff_s']=round(cell_rin['time_unix']-cell_rin['time_unix'].iloc[0]) #add time differecne in seconds 
    cell_rin['time_diff_min']=round(cell_rin['time_diff_s']/60, 1) #and minutes
    
    cell_rin['time_bin_5min'] = (cell_rin['time_diff_s'] // 300) * 300  # 5-minute bins in seconds
    cell_rin['time_bin_5min_label'] = (cell_rin['time_diff_s'] // 300).astype(int)  # 5-minute bin number
    #add the time process the table - only take QC_extended and clamped values, only baseline and refercne condition
    cell_rin_clamped = cell_rin[cell_rin.clamp_mode == 'clamped'] #clamped Rin values
    #add rin_normalisation
    first_rin = cell_rin_clamped['Rin'].iloc[0]
    first_ss_slope = cell_rin_clamped['ss_slope'].iloc[0]
    first_sag_slope = cell_rin_clamped['sag_slope'].iloc[0]
    first_ratio = cell_rin_clamped['ratio_of_slopes'].iloc[0]
    first_rheo = cell_rin_clamped['rheobase'].iloc[0]
    
    cell_rin_clamped.loc[cell_rin_clamped.index, 'Rin_norm'] = (cell_rin_clamped.loc[cell_rin_clamped.index, 'Rin'] / first_rin) 
    cell_rin_clamped.loc[cell_rin_clamped.index, 'ss_slope_norm'] = (cell_rin_clamped.loc[cell_rin_clamped.index, 'ss_slope'] / first_ss_slope) 
    cell_rin_clamped.loc[cell_rin_clamped.index, 'sag_slope_norm'] = (cell_rin_clamped.loc[cell_rin_clamped.index, 'sag_slope'] / first_sag_slope) 
    cell_rin_clamped.loc[cell_rin_clamped.index, 'ratio_of_slopes_norm'] = (cell_rin_clamped.loc[cell_rin_clamped.index, 'ratio_of_slopes'] / first_ratio)
    cell_rin_clamped.loc[cell_rin_clamped.index, 'rheobase_norm'] = (cell_rin_clamped.loc[cell_rin_clamped.index, 'rheobase'] / first_rheo) 
    
    
    
    
    cell_rin_clamped = cell_rin_clamped[cell_rin_clamped.condition.isin(['baseline', 'naag', 'vehicle'])] #only take naag and vehicle now, abandone intermediate steps
    cell_rin_clamped = cell_rin_clamped[cell_rin_clamped['QC_extended'] == 'included'] #only include ones with QC_extended
    
    
    #check if cell has multiple measurements or one
    measurement_counts = cell_rin_clamped.groupby('condition').size()
    has_multiple = any(measurement_counts > 1)
    # Create the measurement_type column
    has_multiple_baseline = False
    if 'baseline' in measurement_counts.index:
        has_multiple_baseline = measurement_counts['baseline'] > 1
        
    cell_rin_clamped['measurement_type'] = 'multiple' if has_multiple else 'single'
    cell_rin_clamped['multiple_baseline'] = 'multiple' if has_multiple_baseline else 'single'
    del has_multiple
    #all the data there is
    rin_total=pd.concat([rin_total, cell_rin_clamped], ignore_index=True)
    

del cell, first_rin, first_ss_slope, first_sag_slope, first_ratio, first_rheo, species

#%% add events to match the timing fo the singels
rin_total_multiple_values=rin_total[rin_total.measurement_type == 'multiple']
target_times = {'baseline': 121,'vehicle': 734, 'naag': 1481}
rin_multiple_timematched = pd.DataFrame()
for (cell, condition), group_data in rin_total_multiple_values.groupby(['cell', 'condition']):
    if condition not in target_times:
        continue
    if len(group_data) == 0:
        continue
    condition_target_time = target_times[condition]
    group_data = group_data.copy()
    group_data['time_difference'] = abs(group_data['time_diff_s'] - condition_target_time)
    closest_row_idx = group_data['time_difference'].idxmin()
    closest_row = group_data.loc[[closest_row_idx]]
    # Add to results
    rin_multiple_timematched = pd.concat([rin_multiple_timematched, closest_row], ignore_index=True)
    
del target_times, closest_row_idx, closest_row, group_data, condition_target_time

rin_multiple_timematched['suitable'] = rin_multiple_timematched['time_difference'] < 120
cell_suitability = rin_multiple_timematched.groupby('cell')['suitable'].all().reset_index()
cell_suitability.rename(columns={'suitable': 'all_conditions_suitable'}, inplace=True)

# Merge this information back to the original DataFrame
rin_multiple_timematched = rin_multiple_timematched.merge(cell_suitability, on='cell',  how='left')

del cell, condition


rin_multiple_timematched.to_excel(f'{savepath}/Statistics/Suppl Figure Chapter2/tables/{current_date}rin_multipletimematched2min.xlsx')

#%% trying to check if we can find a pair of conditions in others cells which we could add to the data, the ones which are not true for both.
rin_single=rin_total[rin_total.measurement_type == 'single']

#get the time differnce I aim aiming for
for cell in rin_single['cell'].unique():
    # Get data for this specific cell
    cell_data = rin_single[rin_single.cell == cell]

    # Get baseline time for this cell
    baseline_times = cell_data[cell_data.condition == 'baseline']['time_unix'].values

    if len(baseline_times) > 0:
        baseline_time = baseline_times[0]  # Use the first baseline measurement as reference

        # Calculate time difference for each non-baseline condition for this cell
        for condition in ['naag', 'vehicle']:
            condition_rows = rin_single[(rin_single.cell == cell) & (rin_single.condition == condition)]
            for idx in condition_rows.index:
                time_diff = rin_single.loc[idx, 'time_unix'] - baseline_time
                rin_single.loc[idx, 'time_from_baseline_s'] = time_diff
                rin_single.loc[idx, 'time_from_baseline_min'] = round(time_diff / 60, 1)

        # Create bins for the time differences (5-minute bins) for this cell
        condition_indices = rin_single[(rin_single.cell == cell) & (rin_single.condition.isin(['naag', 'vehicle']))].index
        if not condition_indices.empty:
            rin_single.loc[condition_indices, 'time_from_baseline_bin_5min'] = (rin_single.loc[condition_indices, 'time_from_baseline_s'] // 300) * 300

rin_single.to_excel(f'{savepath}/Statistics/Suppl Figure Chapter2/tables/{current_date}rin_singles.xlsx')

median_time_vehicle=np.median(rin_single[(rin_single.exp_type == 'control')& ((~rin_single.time_from_baseline_s.isna()))]['time_from_baseline_s'])
median_time_naag=np.median(rin_single[(rin_single.exp_type == 'experiment')& ((~rin_single.time_from_baseline_s.isna()))]['time_from_baseline_s'])


#search for a combination of conditions where the time difference matches
cells_to_process = rin_multiple_timematched[rin_multiple_timematched['all_conditions_suitable']==False]['cell'].unique() #ones where I did not find time matched conditions
result_rows = []  # To store the results for later addition
    
for cell in cells_to_process:
    cell_data = rin_total_multiple_values[rin_total_multiple_values.cell == cell]
