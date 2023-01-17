# Pickle reader
#
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import math
import uproot3 as uproot
from scipy.signal import find_peaks
import numpy.ma as ma
import scipy.stats as ss

def getRootK(FILE):
    global number_events, g4_nParticles, g4_trkID, g4_PDG, nu_isp_number
    global Edep_number, Edep_dq, Edep_x
    global Edep_z, Edep_trkID, nu_isp_pdg, Edep_y, Edep_t,Edep_t_end,Edep_length
    
    with uproot.open(FILE) as f:
        tree = f['event_tree']
        Edep_x = tree.array(['hit_start_x'])
        g4_nParticles = tree.array(['number_particles'])
        g4_trkID = tree.array(['particle_track_id'])
        g4_PDG = tree.array(['particle_pdg_code'])
        Edep_number = tree.array(['number_hits'])
        Edep_dq = tree.array(['hit_energy_deposit'])
        
        Edep_x = tree.array(['hit_start_x'])
        Edep_y = tree.array(['hit_start_y'])
        Edep_z = tree.array(['hit_start_z'])
        
        Edep_t = tree.array(['hit_start_t'])
        Edep_trkID = tree.array(['hit_track_id'])
        Edep_t_end = tree.array(['hit_end_t'])
        Edep_length = tree.array(['hit_length'])

def getParticles(Event, PID):    
    b = np.array([g4_PDG[Event][np.in1d(g4_trkID[Event],Edep_trkID[Event][oo])][0] for oo in range(len(Edep_trkID[Event]))])

    return np.where(b == PID)[0].tolist()

# def update(frame):
#     p = aaa[frame,:,:]
#     ax.imshow(p,interpolation='none',extent=[0,200,200,0])
#     ax.set_title("Time: {}-{} ns".format(frame,frame+1), fontsize=20)
#     ax.set_xlabel('z [cm]', fontsize=15)
#     ax.set_ylabel('y [cm]', fontsize=15)

def getPixleMap(file):
    with open(file, 'rb') as f:
        a = pickle.load(f)
    return a

DUMP = '/Users/elenag/Documents/Papers/LILAr/QpLSimulation/dump'

getRootK('bangbang/banger_100MeVKE.root') 
# Read list of events from file
#with open('Kdecay2Mu_shorter.txt') as f:
with open('Kdecay2Mu.txt') as f:
#with open('Kdecay2Hadrons.txt') as f:
#with open('Kdecay.txt') as f:
#with open('Kdecay2Other.txt') as f:
    lines = f.read().splitlines()
lines = [int(i) for i in lines]
EventList = np.array(lines)


#EventList = np.array([2])

# for i in range(len(EventList)):
#     idx = EventList[i]
#     aaa = getPixleMap(DUMP + '/' + str(idx) + '.pkl')
#     y = np.array([sum(sum(aaa[i])) for i in range(len(aaa))])
#     x = np.arange(len(aaa))
#     plt.plot(x,y)
#     m = (x > 0) & (x < 100)
#     peaks, _ = find_peaks(y[m], width=1.4) 
#     plt.plot(x[m][peaks],y[m][peaks],'or')

# EventList = np.array([959])
# plt.yscale('log')
# idx = 458
tt = []
height = []
y_list = []
y_tot = np.zeros(2000)
identifiedEvent = 0

fig, axs = plt.subplots(2)
fig.suptitle('Plots')



for i in range(len(EventList)):
    idx = EventList[i]
    # mmm = getPixleMap(DUMP + '/mu/' + str(idx) + '.pkl')
    # my = np.array([sum(sum(mmm[i])) for i in range(len(mmm))])
    # mx = np.arange(len(mmm))
    # plt.plot(mx,my,'k')

    # kkk = getPixleMap(DUMP + '/k/' + str(idx) + '.pkl')
    # ky = np.array([sum(sum(kkk[i])) for i in range(len(kkk))])
    # kx = np.arange(len(kkk))
    # plt.plot(kx,ky,'g')

    
    # aaa is the read out pickle
    aaa = getPixleMap(DUMP + '/' + str(idx) + '.pkl')
    #Elena Checking Stuff print(type(aaa), aaa.shape)
    # y is the sum of the photon per 1 ns frame
    y = np.array([sum(sum(aaa[i])) for i in range(len(aaa))])
    y_tot += y
    x = np.arange(len(aaa))
    #axs[0].plot(x, y, '--k') #plt.plot(x,y,'--k')
    #axs[0].set_xlim([0, 200])
    #axs[0].set_title('K --> other dk channels timing')
    #axs[0].set_title('K --> pions timing')
    #axs[0].set_title('K --> mu timing')
    #axs[0].set_xlabel('Time (ns)')
    
    #this is the peak finder
    m = (x > 0) & (x < 100)
    peaks, _ = find_peaks(y[m], width=1.4)
    #Elena Checking Stuff print("How many peaks:", len(peaks))
    if len(peaks) > 1:
        mask = np.ones_like(peaks)
        diff = np.diff(peaks)
        selectedEvents =  diff[ (diff > 4) & (diff < 50)]
        if len(selectedEvents) > 0:
            identifiedEvent += 1
        dt = x[m][peaks[-1]] - x[m][peaks[-2]]
        peakHeight = y[m][peaks[-2]]
        #if peakHeight > y[m][peaks[-2]]:
        #    peakHeight = y[m][peaks[-2]]

        #axs[0].plot(x[m][peaks],y[m][peaks],'or', label = 'Peak Finder')#, label = '%s ns'%dt)
        tt.append(dt)
        height.append(peakHeight)
        #input()
        #mdt = mx[np.where(my == max(my))[0]][0] - kx[np.where(ky == max(ky))[0]][0] 
    
        #kaons = getParticles(idx,321)
        #muons = getParticles(idx,-13)
        #print(kaons,muons)
        #print(idx,dt)

        #break
    
        
    
    #print(idx,dt, mdt)#, round(sum(Edep_dq[idx][kaons])),round(sum(Edep_dq[idx][muons])))


axs[0].hist(height,35)
axs[0].set_title('Photon distribution for lowest peak')
axs[0].set_xlabel('# photons')
    
axs[1].hist(tt,35)
axs[1].set_title('First 2 Peaks Time Diff')
axs[1].set_xlabel('Delta t (ns)')

print(identifiedEvent, "Efficiency: ", float(identifiedEvent)/float(len(EventList)))

plt.show()
#input()

# plt.xlim(0,100) 

# data = np.array(tt)
# m = data > 10
# data = data[m]

# from scipy.optimize import curve_fit

# def func(x,a,b):
#     y = a*np.exp(-b*x)
#     return y

# P = ss.expon.fit(data)
# rX = np.linspace(min(data), max(data), 50)
# rP = ss.expon.pdf(rX, *P)
# plt.hist(data,density = True, color='slategrey')

# plt.plot(rX, rP, color='darkturquoise')
# print(P)





