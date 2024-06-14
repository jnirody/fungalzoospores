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
from scipy.stats import mode, entropy, iqr
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
def get_vel(x,y,t):
    
    dx = np.array(x[1:]) - np.array(x[:-1])
    dy = np.array(y[1:]) - np.array(y[:-1])
    dt = np.array(t[1:]) - np.array(t[:-1])
    
    vx = dx / dt
    vy = dy / dt
    
    vel_vec = [np.array([vx[i], vy[i]]) for i in range(len(vx))]
    
    return vel_vec
    
###########################################################################
def getAngle(vel_vec):

    angle = []
    for i in range(len(vel_vec)-1):
        vt = vel_vec[i]
        v_dt = vel_vec[i+1]
        v_dot = np.dot(vt,v_dt)
        v_cross = np.cross(vt,v_dt)
        theta = abs(np.arctan2(v_cross,v_dot))
        angle.append(theta)
        
    return angle
###########################################################################
warnings.filterwarnings('ignore')
upperdir = '/'.join(os.getcwd().split('/')[:-1])
lowerdir = upperdir + '/Tracks/'
#conditions = glob.glob(lowerdir + '*/')
condition = lowerdir + 'Control/'
folders = (glob.glob(condition + '*/'))

datadic = {}
species = []
if condition.split('/')[-2] == 'Control':
    group1 = ['Allomyces_macrogynus','Catenophlyctis_sp','Allomyces_reticulatus','Blastocladiella_emersonii']
    group2 = ['Clydeaea_visicula','Rhizophlyctis_rosea','Spizellomyces_punctatus','Geranomyces_variabilis']
    group3 = ['Chytriomyces_confervae','Homolaphylyctis_polyrhiza','Rhizoclosmatium_globosum','Synchyrtium_microbalum']
if condition.split('/')[-2] == 'Drugs':
    group1 = ['Taxol_Blastocladiella_emersonii', 'Taxol_Rhizoclosmatium_globosum', 'Taxol_Spizellomyces_punctatum']
    group2 = ['Nocodazole_Blastocladiella_emersonii', 'Nocodazole_Rhizoclosmatium_globusum', 'Nocodazole_Spizellomyces_punctatum']
    group3 = ['Blastocladiella_emersonii', 'Rhizoclosmatium_globosum' ,'Spizellomyces_punctatus']
    sgroup1 = ['Taxol_Blastocladiella_emersonii', 'Nocodazole_Blastocladiella_emersonii', 'Blastocladiella_emersonii']
    sgroup2 = ['Taxol_Rhizoclosmatium_globosum', 'Nocodazole_Rhizoclosmatium_globusum', 'Rhizoclosmatium_globosum']
    sgroup3 = ['Taxol_Spizellomyces_punctatum', 'Nocodazole_Spizellomyces_punctatum', 'Spizellomyces_punctatus']
    
for folder in folders:
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]] = {}
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['times'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['pos'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['scaled_pos'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['angle'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['speed'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['MSD'] = [[]]
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['mean_MSD'] = []
    datadic[folder.split('/')[-2]+'_'+folder.split('/')[-3]]['curvature'] = []
    species.append(glob.glob(folder + '*.csv'))
curvature = [[] for folder in folders]
meanc = [[] for folder in folders]

if condition.split('/')[-2] == 'Control':
    mags = [20,10,10,20,10,10,40,10,10,10,10,10]
if condition.split('/')[-2] == 'Drugs':
    mags = [10,10,10,10,10,10,10,10,10]
mags = [0.84*mag for mag in mags]
for sp in range(len(species)):
    datadic[list(datadic)[sp]]['MSD'].append([])
    for file in species[sp]:
        curr_data = pd.read_csv(file)
        track_names = curr_data['TRACK_ID'].unique().tolist()
        for i in range(len(track_names)):
            times = curr_data[curr_data['TRACK_ID']==track_names[i]]['POSITION_T'].tolist()
            times = [i for i in times]
            if len(times) < 200:
                continue
            if len(times) > len(datadic[list(datadic)[sp]]['MSD'][-1]):
                add_on =  len(times) - len(datadic[list(datadic)[sp]]['MSD'][-1])
                for j in range(add_on):
                    datadic[list(datadic)[sp]]['MSD'][-1].extend([[]])
            datadic[list(datadic)[sp]]['times'].append(times)
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
                    datadic[list(datadic)[sp]]['MSD'][-1][shift].extend([(r[time+shift] - r[time])**2])
            speed = []
            vel_vec = get_vel(x_pos, y_pos, times)
            angle = getAngle(vel_vec)
            for j in range(10,len(x_pos)):
                speed.extend([np.sqrt((x_pos[j]-x_pos[j-1])**2 + (y_pos[j]-y_pos[j-1])**2)/(times[j]-times[j-1])])
                #angle.extend([getAngle(np.array([x_pos[j],y_pos[j]]),np.array([x_pos[j-2],y_pos[j-2]]),np.array([x_pos[j-4],y_pos[j-4]]))])
            datadic[list(datadic)[sp]]['pos'].append([(x_pos[t],y_pos[t]) for t in range(len(x_pos))])
            (x_smooth,y_smooth) = (x_pos,y_pos) #signal.savgol_filter((x_pos,y_pos), window_length=5, polyorder=3, mode = 'nearest')
            smtuples = [(x_smooth[i],y_smooth[i]) for i in range(0,len(x_smooth))]
            path = np.array(datadic[list(datadic)[sp]]['pos'][-1])
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
            datadic[list(datadic)[sp]]['curvature'].extend(data.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5').tolist())
            data['curvature'] = data.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
            curvature[sp].extend(data.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5').tolist())
            data['curvature'] = data.curvature.rolling(4,center=True).mean()
#            c2 = data.curvature.rolling(20,center=True).std()
            curvature[sp].extend(data['curvature'])
            meanc[sp].extend([iqr(angle, nan_policy='omit')])
            datadic[list(datadic)[sp]]['scaled_pos'].append([(x_pos[t]-x_start,y_pos[t]-y_start) for t in range(len(x_pos))])
            datadic[list(datadic)[sp]]['speed'].append(speed)
            datadic[list(datadic)[sp]]['angle'].append(angle)
            datadic[list(datadic)[sp]]['MSD'].extend([[]])
            mean_MSD = (zeros(len(datadic[list(datadic)[sp]]['MSD'][-2])))
            for i in range(len(mean_MSD)):
                mean_MSD[i] = mean(datadic[list(datadic)[sp]]['MSD'][-2][i])
           # plt.loglog(mean_MSD[:400],color='grey')
           # plt.show()


npr = 4
# plot all tracks as subplots
for sp in range(len(list(datadic))):
    total_tracks = len(datadic[list(datadic)[sp]]['times'])
    #print(total_tracks)
    rows = total_tracks // npr
    if total_tracks % npr > 0:
        rows += 1
    fig, axs = plt.subplots(rows,npr)
    for i in range(rows):
        for j in range(npr):
            if i*npr+j < total_tracks:
                path = np.array([*datadic[list(datadic)[sp]]['pos'][i*npr+j]])
#                FFT = np.fft.fftn(datadic[list(datadic)[sp]]['angle'][i*npr+j])
#                PS = abs(FFT)**2
#                plt.plot(PS)
#                plt.show()
                axs[i,j].plot(*path.T)
            axs[i,j].set_yticks([])
            axs[i,j].set_xticks([])
    fig.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_tracks.pdf')

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

# plot track overlays
for sp in range(len(list(datadic))):
    total_tracks = len(datadic[list(datadic)[sp]]['times'])
    compfig = plt.figure()
    for i in range(total_tracks):
        path = np.array([*datadic[list(datadic)[sp]]['scaled_pos'][i]])
        plt.plot(*path.T)
    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_compositetracks.pdf')


meanofmeans_MSD = []
scaled_meanofmeans_MSD = []
for sp in range(len(list(datadic))):
    mean_MSD = []
    std_MSD = []
    fig = plt.subplots()
    for tr in range(len(datadic[list(datadic)[sp]]['MSD'])):
        mean_MSD.append(zeros(len(datadic[list(datadic)[sp]]['MSD'][tr])))
        std_MSD.append(zeros(len(datadic[list(datadic)[sp]]['MSD'][tr])))
        for i in range(len(mean_MSD[-1])):
            mean_MSD[-1][i] = mean(datadic[list(datadic)[sp]]['MSD'][tr][i])#/mean(datadic[list(datadic)[sp]]['MSD'][tr][1])
            std_MSD[-1][i] = std(datadic[list(datadic)[sp]]['MSD'][tr][i])
        plt.loglog(mean_MSD[-1][:400], alpha=8/len(datadic[list(datadic)[sp]]['MSD']),color='grey')
    meanofmeans_MSD.append( np.nanmean(np.array(list(itertools.zip_longest(*mean_MSD)),dtype=float),axis=1))
    temp_scale = [i/meanofmeans_MSD[-1][1] for i in meanofmeans_MSD[-1]]
    scaled_meanofmeans_MSD.append(temp_scale)
    plt.loglog(np.linspace(0,400/60.,400),meanofmeans_MSD[sp][:400],color='red')
    plt.xlim([1,1e3])
    plt.ylim([1e-2,1e6])
   # plt.loglog(std_MSD[:400])
    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_MSD.pdf')
#

fig = plt.subplots()
group1means = []
group2means = []
group3means = []
group1_meanofmeans_MSD = []
group2_meanofmeans_MSD = []
group3_meanofmeans_MSD = []
if condition.split('/')[-2] == 'Control':
    labels = ['Blastocladiomycota', 'Chytridiomycota - tubulin', 'Chytridiomycota - actin']
if condition.split('/')[-2] == 'Drugs':
    labels = ['Taxol', 'Nocodazole', 'No Drug']
for sp in range(len(list(datadic))):
    if species[sp][0].split('/')[-2] in group1:
        group1means.append(scaled_meanofmeans_MSD[sp][:400])
        plt.loglog(np.linspace(0,400/60.,400),scaled_meanofmeans_MSD[sp][:400],alpha=0.2,color='green')
    if species[sp][0].split('/')[-2] in group2:
        group2means.append(scaled_meanofmeans_MSD[sp][:400])
        plt.loglog(np.linspace(0,400/60.,400),scaled_meanofmeans_MSD[sp][:400],alpha=0.2,color='purple')
    if species[sp][0].split('/')[-2] in group3:
        group3means.append(scaled_meanofmeans_MSD[sp][:400])
        plt.loglog(np.linspace(0,400/60.,400),scaled_meanofmeans_MSD[sp][:400],alpha=0.2,color='darkorange')
group1_meanofmeans_MSD = np.nanmean(np.array(list(itertools.zip_longest(*group1means)),dtype=float),axis=1)
group2_meanofmeans_MSD =  np.nanmean(np.array(list(itertools.zip_longest(*group2means)),dtype=float),axis=1)
group3_meanofmeans_MSD = np.nanmean(np.array(list(itertools.zip_longest(*group3means)),dtype=float),axis=1)
plt.loglog(np.linspace(0,400/60.,400),group1_meanofmeans_MSD[:400],color='green',label=labels[0],linewidth=2)
plt.loglog(np.linspace(0,400/60.,400),group2_meanofmeans_MSD[:400],color='purple',label=labels[1],linewidth=2)
plt.loglog(np.linspace(0,400/60.,400),group3_meanofmeans_MSD[:400],color='darkorange', label=labels[2],linewidth=2)
plt.legend()
plt.savefig(upperdir + '/Plots/' + condition.split('/')[-2] + '_allspecies_MSD.pdf')

fig = plt.subplots()
group1means = []
group2means = []
group3means = []
labels = ['Blastocladiomycota', 'Chytridiomycota - tubulin', 'Chytridiomycota - no tubulin']
for sp in range(len(list(datadic))):
    print(list(datadic)[sp])
    merged = list(itertools.chain(*datadic[list(datadic)[sp]]['angle']))
    if species[sp][0].split('/')[-2] in group1:
        group1means.extend(merged)
        sns.kdeplot(merged, alpha = 0.2, color = 'green')
    if species[sp][0].split('/')[-2] in group2:
        group2means.extend(merged)
        sns.kdeplot(merged, alpha = 0.2, color = 'purple')
    if species[sp][0].split('/')[-2] in group3:
        group3means.extend(merged)
        sns.kdeplot(merged, alpha = 0.2, color = 'darkorange')
sns.kdeplot(group1means, color = 'green', label=labels[0], linewidth = 2)
sns.kdeplot(group2means, color = 'purple', label=labels[1], linewidth = 2)
sns.kdeplot(group3means, color = 'darkorange', label=labels[2], linewidth = 2)
plt.xticks([0,math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
plt.legend()
plt.savefig(upperdir + '/Plots/' +  'allspecies_angles.pdf')


if condition.split('/')[-2] == 'Drugs':
    sgroup1means = []
    sgroup2means = []
    sgroup3means = []
    sgroup1_meanofmeans_MSD = []
    sgroup2_meanofmeans_MSD = []
    sgroup3_meanofmeans_MSD = []
    fig = plt.subplots()
    for sp in range(len(list(datadic))):
        if species[sp][0].split('/')[-2] in sgroup1:
            sgroup1means.append(scaled_meanofmeans_MSD[sp][:400])
            if species[sp][0].split('/')[-2] in group1:
                clabel = labels[0]
            if species[sp][0].split('/')[-2] in group2:
                clabel = labels[1]
            if species[sp][0].split('/')[-2] in group3:
                clabel = labels[2]
            plt.loglog(np.linspace(0,800/60.,400),scaled_meanofmeans_MSD[sp][:400],label = clabel)
   # sgroup1_meanofmeans_MSD = np.nanmean(np.array(list(itertools.zip_longest(*sgroup1means)),dtype=float),axis=1)
    #plt.loglog(sgroup1_meanofmeans_MSD[:400], color='green', linewidth=2)
    plt.legend()
    plt.savefig(upperdir + '/Plots/' + sgroup1[2] + '_drugcomp_MSD.pdf')
    fig = plt.subplots()
    for sp in range(len(list(datadic))):
        if species[sp][0].split('/')[-2] in sgroup2:
            sgroup2means.append(scaled_meanofmeans_MSD[sp][:400])
            if species[sp][0].split('/')[-2] in group1:
                clabel = labels[0]
            if species[sp][0].split('/')[-2] in group2:
                clabel = labels[1]
            if species[sp][0].split('/')[-2] in group3:
                clabel = labels[2]
            plt.loglog(np.linspace(0,800/60.,400),scaled_meanofmeans_MSD[sp][:400], label=clabel)
    #sgroup2_meanofmeans_MSD =  np.nanmean(np.array(list(itertools.zip_longest(*sgroup2means)),dtype=float),axis=1)
   # plt.loglog(sgroup2_meanofmeans_MSD[:400], color='purple', linewidth=2)
    plt.legend()
    plt.savefig(upperdir + '/Plots/' + sgroup2[2] + '_drugcomp_MSD.pdf')
    fig = plt.subplots()
    for sp in range(len(list(datadic))):
        if species[sp][0].split('/')[-2] in sgroup3:
            sgroup3means.append(scaled_meanofmeans_MSD[sp][:400])
            if species[sp][0].split('/')[-2] in group1:
                clabel = labels[0]
            if species[sp][0].split('/')[-2] in group2:
                clabel = labels[1]
            if species[sp][0].split('/')[-2] in group3:
                clabel = labels[2]
            plt.loglog(np.linspace(0,800/60.,800),scaled_meanofmeans_MSD[sp][:800], label = clabel)
    #sgroup3_meanofmeans_MSD = np.nanmean(np.array(list(itertools.zip_longest(*sgroup3means)),dtype=float),axis=1)
    #plt.loglog(sgroup3_meanofmeans_MSD[:400], color='darkorange' ,linewidth=2)
    plt.legend()
    plt.savefig(upperdir + '/Plots/' + sgroup3[2] + '_drugcomp_MSD.pdf')

##calculate instantaneous speeds
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
#                print(len(temp_speeds))
#                speeds = [x for x in temp_speeds if not math.isnan(x) and not math.isinf(x)]
#                times = [temp_times[t] for t in range(min(len(temp_times),len(temp_speeds))) if not math.isnan(temp_speeds[t]) and not math.isinf(temp_speeds[t])]
#                axs[i,j].plot(times,speeds)
#            axs[i,j].set_yticks([])
#            axs[i,j].set_xticks([])
#    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_speeds.pdf')

compiled_speeds = []
splabels = []
for sp in range(len(list(datadic))):
    fig, ax = plt.subplots()
    splabels.append(list(datadic)[sp].split('_')[0][0] + list(datadic)[sp].split('_')[1][0])
    for track in range(10):
        plt.plot(datadic[list(datadic)[sp]]['speed'][track], alpha = 0.2)
    temp_speeds = [int_speed for speeds in datadic[list(datadic)[sp]]['speed'] for int_speed in speeds]
    compiled_speeds.append([x for x in temp_speeds if not math.isnan(x) and not math.isinf(x)])
    plt.tight_layout()
    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_speeds.pdf')
with open('rawspeeds.csv', 'w') as f:
    writer = csv.writer(f)
    for sp in range(len(compiled_speeds)):
        writer.writerow(compiled_speeds[sp])
fig, ax = plt.subplots()
g = sns.violinplot(data=compiled_speeds)
g.set_xticklabels(splabels)
#ax.set_xticklabels([list(datadic)[0],list(datadic)[1],list(datadic)[2],list(datadic)[3],list(datadic)[4],list(datadic)[5],list(datadic)[6]])
#ax.set_ylabel('Instantaneous speed (pixels/frame)')
#plt.tight_layout()
plt.savefig(upperdir + '/Plots/' + condition.split('/')[-2] + '_Speed_Comparison.pdf')

compiled_angles = []
splabels = []
for sp in range(len(list(datadic))):
    fig, ax = plt.subplots()
    splabels.append(list(datadic)[sp].split('_')[0][0] + list(datadic)[sp].split('_')[1][0])
    for track in range(10):
        sns.kdeplot(data=datadic[list(datadic)[sp]]['angle'][track])
    temp_angles = [int_ang for angles in datadic[list(datadic)[sp]]['angle'] for int_ang in angles]
    compiled_angles.append([x for x in meanc[sp] if not math.isnan(x) and not math.isinf(x)])
    plt.tight_layout()
    plt.savefig(upperdir + '/Plots/' + list(datadic)[sp] + '_angles.pdf')
fig, ax = plt.subplots()
g = sns.violinplot(data=compiled_angles)
g.set_xticklabels(splabels)
#ax.set_xticklabels([list(datadic)[0],list(datadic)[1],list(datadic)[2],list(datadic)[3],list(datadic)[4],list(datadic)[5],list(datadic)[6]])
#ax.set_ylabel('Instantaneous speed (pixels/frame)')
#plt.tight_layout()
plt.savefig(upperdir + '/Plots/' + condition.split('/')[-2] + '_Angle_Comparison.pdf')
