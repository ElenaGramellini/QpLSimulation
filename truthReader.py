# I think this one reads the root file and does some sort of filtering
import pandas as pd
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



######################################## Main ##########################################
HEAD_ROOT = ''
DUMP = '.'

#getRootK('banger_0_31.root')
getRootK('bangbang/banger_100MeVKE.root')

treeName   = "event_tree"
inFileName = "banger_0_31.root"
inFilePath = ""


data      = uproot.open(inFilePath+inFileName)[treeName]
variables = ["run","event","generator_initial_particle_energy", "generator_initial_particle_pdg_code", "particle_pdg_code","number_particles","hit_track_id","number_hits","particle_track_id","particle_initial_energy","particle_parent_track_id", "particle_mass","generator_final_particle_pdg_code"]
#make the list unique
variables = list(set(variables))    
data = data.pandas.df(variables, flatten=False)
#print(data["number_particles"].idxmin())
#print(type(data["hit_track_id"][0]), (data["hit_track_id"][0]).shape,data["number_particles"][0],data["number_hits"][0] )

#Messing around with track ID
#print(type(data["hit_track_id"][443]), (data["hit_track_id"][443]).shape,data["number_particles"][443],data["number_hits"][443] )

#for j in range(data["number_particles"][443]):
#    print(data["particle_parent_track_id"][443][j], data["particle_pdg_code"][443][j], data["particle_track_id"][443][j], data["particle_initial_energy"][443][j] - data["particle_mass"][443][j])



'''
print("Hits             ", type(data["hit_track_id"][443]), data["hit_track_id"][443].shape)
print("Particle         ",type(data["particle_parent_track_id"][443]), data["particle_parent_track_id"][443].shape)
print("Initial Particle ", type(data["generator_initial_particle_energy"][443]), data["generator_initial_particle_energy"][443].shape)

print( data["particle_parent_track_id"][443].shape, data["particle_pdg_code"][443].shape, data["particle_initial_energy"][443].shape)
'''




#df = pd.DataFrame(np.concatenate([data["particle_parent_track_id"][443], data["particle_pdg_code"][443], data["particle_initial_energy"][443]], axis=0), columns= ['p_Parent','p_Pdg','p_Energy'])

def makeParticleDf(var0, var1, var2,var3):
    df = pd.DataFrame(np.vstack([var0, var1, var2, var3]).T, columns= ['p_Parent','p_Pdg','p_Energy','p_ID'])
    #df = pd.DataFrame(np.vstack([var0, var1, var2]).T, columns= ['p_Parent','p_Pdg','p_Energy'])
    df['p_Pdg'] = df['p_Pdg'].astype(int)
    kaonID  = df.loc[ (df['p_Pdg'] == 321) ] ['p_ID']
    #print(int(max(kaonID)))
    
    puppa  = df.loc[ (df['p_Parent'] == int(max(kaonID)) ) & (df['p_Energy'] > 0.)] ['p_Pdg']
    #print(set(puppa))
    if sum(puppa.isin([-13])):
        if sum(puppa.isin([111])):
            return 4 # mu+ pi0 3.3% 
        else:
            print(puppa)
            return 2 # mu+ 63.6%
        
    if sum(puppa.isin([211])):
        if sum(puppa.isin([111])):
            return 9 #pi+ pi0 # 20.7% or pi+ pi0 pi0 1.8%
        elif sum(puppa.isin([-211])) :
            return 10 # pi+ pi+ pi- 5.6% 
    if sum(puppa.isin([111])):
        return 3 #pi0 e+ 5.0 %

    if sum(puppa.isin([310])) + sum(puppa.isin([130])):
        return -1 #float('NaN') #this is a K0L or K0S not sure that he's doing here....

    print(set(puppa))
    return float('NaN')


data['decayCode'] = data.apply(lambda x: makeParticleDf(x['particle_parent_track_id'], x['particle_pdg_code'], x['particle_initial_energy'],x['particle_track_id']), axis =1  )

#
muonChannel  = data.loc[ (data['decayCode'] != -1) & (data['decayCode'] != 9) & (data['decayCode'] != 2) ] # code 2 for actual muon channel 
#f = open("Kdecay2Mu.txt", "w")
#f = open("Kdecay2Hadrons.txt", "w")
#f = open("KInteractions.txt", "w")
f = open("Kdecay2Other.txt", "w")
f.write(muonChannel['event'].to_string(index=False))
f.close()

fig, ax = plt.subplots()
labels = ["K0L", "e","","mu+ nu nu", "pi0 e+","pi0 mu+","","","","","pi+pi0","pi+pi+pi-" ]
plottingData = data["decayCode"].dropna() # Eliminate interactions which were flagged as NaN
print(plottingData)
n, bins, patches = ax.hist(plottingData, 11, density=True, facecolor='g', alpha=0.75)
#for i in range(len(n)):
print(sum(n))
print (n/sum(n))
print (labels)
#print(n/sum(n))
ax.set_xticks(bins)
ax.set_xticklabels(labels, rotation='vertical', fontsize=10)
ax.set_title('Final State after last 321 particle')
plt.show()
