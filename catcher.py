# I think this one reads the root file and does some sort of filtering
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
    
    with uproot.open(HEAD_ROOT + FILE) as f:
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
# HEAD = 'C:\\Users/msroo/My Drive/UTA/Physics_UTA/PhD/Simulation/data/'
HEAD_ROOT = ''
DUMP = '.'
# getRootK('banger_iso.root')
getRootK('banger_0_31.root')


# EventList = np.array([4, 17, 59, 85, 103, 172, 191, 192, 215, 252, 259, 260, 263, 270, 287, 
#                       320, 344, 352, 366, 399, 402, 470, 480, 496, 545, 565, 585, 598, 621, 
#                       634, 637, 676, 699, 705, 793, 821, 834, 842, 843, 861, 866, 875, 880, 
#                       893, 902, 924, 959, 962, 997])
# print(len(EventList))

# 

EventList = EventList_0_31 = np.array([28, 105,243,290,317,358,364,390,394,473,479,621,656,667,669,801,898,905])
# EventList = EventList_1_31 = np.array([959,393,458,522,532,500,578,343,822,872,907,959,289,228,73,158,29,2])
# EventList = EventList_2_31 = np.array([970, 650, 460,784, 901, 911,931,698,384,182,163,149,242,256,272,77])
# EventList = EventList_3_31 = np.array([990, 978, 972, 954, 925, 911, 855, 703, 681,570, 562, 561, 472, 520, 509, 390,188,216,223, 16,387,14 ])
# EventList = EventList_4_31 = np.array([989,964,946,905,885,870,865,824,817,774,669,630,562,487,484,375,359,345,228,223,186,178,16])
# EventList = EventList_5_31 = np.array([994,829, 778, 692,672,611,587,514,36,4])
# EventList = EventList_6_31 = np.array([893,875,821,793,705,470,402,352,344,320,287,215,192,191,59,])

# EventList = np.array([25])
# print(len(EventList_6_31))
# idxs = []


'''
tt = []
for i in range(len(EventList)):
    idx = EventList[i]
    aaa = getPixleMap(DUMP + '/' + str(idx) + '.pkl')
    y = np.array([sum(sum(aaa[i])) for i in range(len(aaa))])
    x = np.arange(len(aaa))
    m = (x > 0) & (x < 100)
    peaks, _ = find_peaks(y[m], width=1.4) 
    # plt.figure()
    plt.plot(x,y,'k')
    plt.title('Event %s'%EventList[i])
    plt.plot(x[m][peaks],y[m][peaks],'or')
    tt.append(peaks[-1]-peaks[0])
plt.xlim(0,100)
tt = np.array(tt)
print(tt.tolist())

t = np.array([14, 20, 17, 17, 12, 25, 10, 22, 32, 44, 22, 29, 29, 65, 18,15, 11, 25, 89, 31, 70, 9, 24, 16, 36,
              17, 18, 46, 16, 25, 24, 17, 12, 21, 17, 30, 26, 17, 11, 13, 13, 28, 45, 72, 9, 50, 40, 36, 
              14, 25, 29, 47, 29, 13, 18, 14, 7,32, 11, 18, 9, 20, 11, 24, 15, 9, 37, 10, 29, 47,42, 13, 6, 22, 24, 29, 
              14, 16, 17, 79, 29, 31, 16, 9, 20, 11,38, 28, 49, 22, 14, 66, 25, 26, 14, 23, 13, 38, 46, 17, 10, 15, 11, 29,
              31, 19, 18, 17, 10, 30, 42, 24, 41, 10, 12, 16, 24, 24, 12, 14, 64, 42]) 


'''


