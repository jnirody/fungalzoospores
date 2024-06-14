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
from scipy.stats import mode, entropy
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
from scipy import signal
import matplotlib.gridspec as gridspec
import seaborn as sns
import itertools
import warnings
###########################################################################
def get_smooth(data, smoothing):

    df = pd.DataFrame(data).rolling(smoothing).mean().dropna()

    # first derivatives
    df['dx'] = np.gradient(df[0])
    df['dy'] = np.gradient(df[1])

    df['dx'] = df.dx.rolling(smoothing, center=True).mean()
    df['dy'] = df.dy.rolling(smoothing, center=True).mean()

    # second derivatives
    df['d2x'] = np.gradient(df.dx)
    df['d2y'] = np.gradient(df.dy)

    df['d2x'] = df.d2x.rolling(smoothing, center=True).mean()
    df['d2y'] = df.d2y.rolling(smoothing, center=True).mean()
    
    # calculation of curvature from the typical formula
    df['curvature'] = df.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
    # mask = curvature < 100

    df['curvature'] = df.curvature.rolling(smoothing, center=True).mean()

    df.dropna(inplace=True)
    return df[0], df.curvature
    
###########################################################################
warnings.filterwarnings('ignore')

upperdir = '/'.join(os.getcwd().split('/')[:-1])
lowerdir = upperdir + '/Tracks/'
#conditions = glob.glob(lowerdir + '*/')
comparison = lowerdir + 'Jasplakinolide/'
control = lowerdir + 'Control/'
folders1 = (glob.glob(comparison + '*/'))
folders2 = (glob.glob(control+ '*/'))

datadic = {}
species = [[],[]]
conditions = [comparison.split('/')[-2],control.split('/')[-2]]

for folder in folders1:
    datadic[folder.split('/')[-2]] = {}
    datadic[folder.split('/')[-2]][conditions[0]] = {}
    datadic[folder.split('/')[-2]][conditions[1]] = {}
    datadic[folder.split('/')[-2]][conditions[0]]['times'] = []
    datadic[folder.split('/')[-2]][conditions[0]]['pos'] = []
    datadic[folder.split('/')[-2]][conditions[0]]['scaled_pos'] = []
    datadic[folder.split('/')[-2]][conditions[0]]['angle'] = []
    datadic[folder.split('/')[-2]][conditions[0]]['speed'] = []
    datadic[folder.split('/')[-2]][conditions[0]]['MSD'] = [[]]
    datadic[folder.split('/')[-2]][conditions[0]]['mean_MSD'] = []
    datadic[folder.split('/')[-2]][conditions[0]]['curvature'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['times'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['pos'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['scaled_pos'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['angle'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['speed'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['MSD'] = [[]]
    datadic[folder.split('/')[-2]][conditions[1]]['mean_MSD'] = []
    datadic[folder.split('/')[-2]][conditions[1]]['curvature'] = []
    species[0].append(glob.glob(folder + '*.csv'))
curvature = [[] for folder in folders1]

for folder in folders2:
    if folder.split('/')[-2] in datadic.keys():
        species[1].append(glob.glob(folder + '*.csv'))
    else:
        continue

mags = [10 for i in range(len(species[0]))]
mags = [0.84*mag for mag in mags]
for con in range(len(conditions)):
    for sp in range(len(species[con])):
        datadic[list(datadic)[sp]][conditions[con]]['MSD'].append([])
        for file in species[con][sp]:
            curr_data = pd.read_csv(file)
            track_names = curr_data['TRACK_ID'].unique().tolist()
            for i in range(len(track_names)):
 #               print(file, track_names[i])
                times = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_T'].tolist()
                times = [i for i in times]
                if len(times) < 200:
                    continue
                if len(times) > len(datadic[list(datadic)[sp]][conditions[con]]['MSD'][-1]):
                    add_on =  len(times) - len(datadic[list(datadic)[sp]][conditions[con]]['MSD'][-1])
                    for j in range(add_on):
                        datadic[list(datadic)[sp]][conditions[con]]['MSD'][-1].extend([[]])
                datadic[list(datadic)[sp]][conditions[con]]['times'].append(times)
                x_pos = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_X'].tolist()
                x_pos = [i/mags[sp] for i in x_pos]
                x_start = x_pos[0]
                y_pos = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_Y'].tolist()
                y_pos = [i/mags[sp] for i in y_pos]
                y_start = y_pos[0]
                r = [np.sqrt((x-x_start)**2 + (y-y_start)**2) for (x,y) in zip(x_pos,y_pos)]
                for time in range(int(len(r)*0.7)):
                    for shift in range(int(len(r)*0.7)-time):
                        #if shift == 1:
                       # print((r[time+shift] - r[time])**2)
                        datadic[list(datadic)[sp]][conditions[con]]['MSD'][-1][shift].extend([(r[time+shift] - r[time])**2])
                speed = []
                angle = []
                for j in range(10,len(x_pos)):
                    speed.extend([np.sqrt((x_pos[j]-x_pos[j-10])**2 + (y_pos[j]-y_pos[j-10])**2)/(times[j]-times[j-10])])
                    if x_pos[j] == x_pos[j-10]:
                        #angle.extend([0])
                        continue
                    else:
                        angle.extend([math.atan(abs((y_pos[j]-y_pos[j-10])/(x_pos[j]-x_pos[j-10])))])
                datadic[list(datadic)[sp]][conditions[con]]['pos'].append([(x_pos[t],y_pos[t]) for t in range(len(x_pos))])
                (x_smooth,y_smooth) = (x_pos,y_pos) #signal.savgol_filter((x_pos,y_pos), window_length=5, polyorder=3, mode = 'nearest')
                smtuples = [(x_smooth[i],y_smooth[i]) for i in range(0,len(x_smooth))]
                path = np.array(datadic[list(datadic)[sp]][conditions[con]]['pos'][-1])
                smoothed_path = np.array(smtuples)
                data = pd.DataFrame(smtuples)
                data['dx'] = np.gradient(data[0])
                data['dy'] = np.gradient(data[1])
                #data['dx'] = data.dx.rolling(10,center=True).mean()
                #data['dy'] = data.dy.rolling(10,center=True).mean()
                data['d2x'] = np.gradient(data.dx)
                data['d2y'] = np.gradient(data.dy)
                #data['d2x'] = data.d2x.rolling(10,center=True).mean()
                #data['d2y'] = data.d2y.rolling(10, center=True).mean()
    #            c2 = data.curvature.rolling(20,center=True).std()
                datadic[list(datadic)[sp]][conditions[con]]['scaled_pos'].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
                datadic[list(datadic)[sp]][conditions[con]]['speed'].append(speed)
                datadic[list(datadic)[sp]][conditions[con]]['angle'].append(angle)
                datadic[list(datadic)[sp]][conditions[con]]['MSD'].extend([[]])
                mean_MSD = (zeros(len(datadic[list(datadic)[sp]][conditions[con]]['MSD'][-2])))
                for i in range(len(mean_MSD)):
                    mean_MSD[i] = mean(datadic[list(datadic)[sp]][conditions[con]]['MSD'][-2][i])
 #               plt.loglog(mean_MSD[:400],color='grey')
 #               plt.show()
#npr = 4
## plot all tracks as subplots
#for sp in range(len(list(datadic))):
#    total_tracks = len(datadic[list(datadic)[sp]]['times'])
#    #print(total_tracks)
#    rows = total_tracks // npr
#    if total_tracks % npr > 0:
#        rows += 1
#    fig, axs = plt.subplots(rows,npr)
#    for i in range(rows):
#        for j in range(npr):
#            if i*npr+j < total_tracks:
#                path = np.array([*datadic[list(datadic)[sp]]['pos'][i*npr+j]])
##                FFT = np.fft.fftn(datadic[list(datadic)[sp]]['angle'][i*npr+j])
##                PS = abs(FFT)**2
##                plt.plot(PS)
##                plt.show()
#                axs[i,j].plot(*path.T)
#            axs[i,j].set_yticks([])
#            axs[i,j].set_xticks([])
#    fig.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_tracks.pdf')

#for sp in range(len(list(datadic))):
#    total_tracks = len(datadic[list(datadic)[sp]]['times'])
#    rows = total_tracks // npr
#    if total_tracks % npr > 0:
#        rows += 1
#    fig, axs = plt.subplots(rows,npr)
#    for i in range(rows):
#        for j in range(npr):
#            if i*npr+j < total_tracks:
#               # x_pos = [point[0] for point in datadic[list(datadic)[sp]]['pos'][i*npr+j]]
#               # y_pos = [point[1] for point in datadic[list(datadic)[sp]]['pos'][i*npr+j]]
#               # axs[i,j].plot(range(len(x_pos)),x_pos)
#                axs[i,j].plot(datadic[list(datadic)[sp]]['angle'][i*npr+j])
#            axs[i,j].set_yticks([])
#            axs[i,j].set_xticks([])
#    fig.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_xypositions.pdf')
#
## plot track overlays
#for sp in range(len(list(datadic))):
#    total_tracks = len(datadic[list(datadic)[sp]]['times'])
#    compfig = plt.figure()
#    for i in range(total_tracks):
#        path = np.array([*datadic[list(datadic)[sp]]['scaled_pos'][i]])
#        plt.plot(*path.T)
#    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_compositetracks.pdf')


meanofmeans_MSD = []
scaled_meanofmeans_MSD = []
totaltracks = len(species[0])
if totaltracks < 4:
    npr = totaltracks
elif totaltracks%3 == 0:
    npr = 3
elif totaltracks%4 == 0:
    npr = 4
else:
    npr = 2
rows = totaltracks // npr
fig, axs = plt.subplots(rows, npr)
for condition in conditions:
    for sp in range(len(list(datadic))):
        print(list(datadic)[sp], condition)
        mean_MSD = []
        std_MSD = []
        for tr in range(len(datadic[list(datadic)[sp]][condition]['MSD'])):
            mean_MSD.append(zeros(len(datadic[list(datadic)[sp]][condition]['MSD'][tr])))
            std_MSD.append(zeros(len(datadic[list(datadic)[sp]][condition]['MSD'][tr])))
            for k in range(len(mean_MSD[-1])):
                mean_MSD[-1][k] = mean(datadic[list(datadic)[sp]][condition]['MSD'][tr][k])/mean(datadic[list(datadic)[sp]][condition]['MSD'][tr][1])
                std_MSD[-1][k] = std(datadic[list(datadic)[sp]][condition]['MSD'][tr][k])
#            if condition == 'Control':
#                axs[sp//npr,sp%npr].loglog(mean_MSD[-1][:400], alpha=3/len(datadic[list(datadic)[sp]][condition]['MSD']),color='orange')
#            else:
#                axs[sp//npr,sp%npr].loglog(mean_MSD[-1][:400], alpha=3/len(datadic[list(datadic)[sp]][condition]['MSD']),color='blue')
        meanofmeans_MSD =        np.nanmean(np.array(list(itertools.zip_longest(*mean_MSD)),dtype=float),axis=1)
        temp_scale = [i/meanofmeans_MSD[1] for i in meanofmeans_MSD]
        scaled_meanofmeans_MSD = temp_scale
        if totaltracks > 1:
            axs[sp//npr,sp%npr].loglog(scaled_meanofmeans_MSD[:300])
            axs[sp//npr,sp%npr].title.set_text(list(datadic)[sp].split('_')[0][0] + list(datadic)[sp].split('_')[1][0])
            axs[sp//npr,sp%npr].set_xlim([1,1e3])
            axs[sp//npr,sp%npr].set_ylim([1e-1,1e5])
        else:
            axs.loglog(scaled_meanofmeans_MSD[:300])
            axs.title.set_text(list(datadic)[sp].split('_')[0][0] + list(datadic)[sp].split('_')[1][0])
            axs.set_xlim([1,1e3])
            axs.set_ylim([1e-1,1e5])
plt.tight_layout()
plt.savefig(upperdir + '/Plots/' + conditions[0] + '_comparative_MSD.pdf')
