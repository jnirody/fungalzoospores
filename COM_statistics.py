###########################################################################
#!/usr/bin/python
import re, math, sys, os, random
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob, csv
from scipy.stats import mode
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.gridspec as gridspec
import seaborn as sns
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def gauss(x,mu,sigma,A,c):
    return A*exp(-(x-mu)**2/2/sigma**2) + c
###########################################################################

upperdir = '/'.join(os.getcwd().split('/')[:-1])
lowerdir = upperdir + '/Tracks/'
conditions = glob.glob(lowerdir + '*/')
folders = []
for condition in conditions:
    if condition.split('/')[-2] == 'Control':
        folders.extend(glob.glob(condition + '*/'))

datadic = {}
species = []
for folder in folders:
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]] = {}
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['times'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['pos'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['scaled_pos'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['speed'] = []
    species.append(glob.glob(folder + '*.csv'))

for sp in range(len(species)):
    for file in species[sp]:
        curr_data = pd.read_csv(file)
        track_names = curr_data['TRACK_ID'].unique().tolist()
        for i in range(len(track_names)):
            times = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_T'].tolist()
            datadic[list(datadic)[sp]]['times'].append(times)
            x_pos = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_X'].tolist()
            x_start = x_pos[0]
            y_pos = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_Y'].tolist()
            y_start = y_pos[0]
            speed = []
            for j in range(10,len(x_pos)):
                speed.extend([np.sqrt((x_pos[j]-x_pos[j-10])**2 + (y_pos[j]-y_pos[j-10])**2)/(times[j]-times[j-10])])
            datadic[list(datadic)[sp]]['pos'].append([(x_pos[t],y_pos[t]) for t in range(len(x_pos))])
            datadic[list(datadic)[sp]]['scaled_pos'].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
            datadic[list(datadic)[sp]]['speed'].append(speed)

npr = 6
# plot all tracks as subplots
for sp in range(len(list(datadic))):
    total_tracks = len(datadic[list(datadic)[sp]]['times'])
    rows = total_tracks // npr
    if total_tracks % npr > 0:
        rows += 1
    fig, axs = plt.subplots(rows,npr)
    for i in range(rows):
        for j in range(npr):
            if i*npr+j < total_tracks:
                path = np.array([*datadic[list(datadic)[sp]]['pos'][i*npr+j]])
                axs[i,j].plot(*path.T)
            axs[i,j].set_yticks([])
            axs[i,j].set_xticks([])
    fig.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_tracks.pdf')

for sp in range(len(list(datadic))):
    total_tracks = len(datadic[list(datadic)[sp]]['times'])
    rows = total_tracks // npr
    if total_tracks % npr > 0:
        rows += 1
    fig, axs = plt.subplots(rows,npr)
    for i in range(rows):
        for j in range(npr):
            if i*npr+j < total_tracks:
                x_pos = [point[0] for point in datadic[list(datadic)[sp]]['pos'][i*npr+j]]
                y_pos = [point[1] for point in datadic[list(datadic)[sp]]['pos'][i*npr+j]]
                axs[i,j].plot(range(len(x_pos)),x_pos)
                axs[i,j].plot(range(len(y_pos)),y_pos)
            axs[i,j].set_yticks([])
            axs[i,j].set_xticks([])
    fig.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_xypositions.pdf')

# plot track overlays
for sp in range(len(list(datadic))):
    total_tracks = len(datadic[list(datadic)[sp]]['times'])
    rows = total_tracks // npr
    if total_tracks % npr > 0:
        rows += 1
    compfig = plt.figure()
    for i in range(rows):
        for j in range(npr):
            if i*npr+j < total_tracks:
                path = np.array([*datadic[list(datadic)[sp]]['scaled_pos'][i*npr+j]])
                plt.plot(*path.T)
    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_compositetracks.pdf')
    
# calculate

#calculate instantaneous speeds
#for sp in range(len(list(datadic))):
#    total_tracks = len(datadic[list(datadic)[sp]]['times'])
#    rows = total_tracks // 5
#    if total_tracks % 5 > 0:
#        rows += 1
#    fig, axs = plt.subplots(rows,5)
#    for i in range(rows):
#        for j in range(5):
#            if i*5+j < total_tracks:
#                temp_speeds = datadic[list(datadic)[sp]]['speed'][i*5+j]
#                temp_times = datadic[list(datadic)[sp]]['times'][i*5+j]
#                speeds = [x for x in temp_speeds if not math.isnan(x) and not math.isinf(x)]
#                times = [temp_times[t] for t in range(len(temp_times)-1) if not math.isnan(temp_speeds[t]) and not math.isinf(temp_speeds[t])]
#                axs[i,j].plot(times,speeds)
#            axs[i,j].set_yticks([])
#            axs[i,j].set_xticks([])
#    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_speeds.pdf')

fig, ax = plt.subplots()
compiled_speeds = []
for sp in range(len(list(datadic))):
        print(list(datadic)[sp])
        temp_speeds = [int_speed for speeds in datadic[list(datadic)[sp]]['speed'] for int_speed in speeds]
        compiled_speeds.append([x for x in temp_speeds if not math.isnan(x) and not math.isinf(x)])
with open('rawspeeds.csv', 'w') as f:
    writer = csv.writer(f)
    for sp in range(len(compiled_speeds)):
        print(compiled_speeds[sp])
        writer.writerow(compiled_speeds[sp])
sns.violinplot(data=compiled_speeds,cut=0)
#ax.set_xticklabels([list(datadic)[0],list(datadic)[1],list(datadic)[2],list(datadic)[3],list(datadic)[4],list(datadic)[5],list(datadic)[6]])
#ax.set_ylabel('Instantaneous speed (pixels/frame)')
plt.tight_layout()
plt.savefig(upperdir + '/Plots/' + 'Speed_Comparison.pdf')

