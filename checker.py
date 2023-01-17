import numpy as np
import uproot3 as uproot
import pickle
import csv
import matplotlib.pyplot as plt
import math

def getRoot(FILE):
    global number_events, g4_nParticles, g4_trkID, g4_PDG, nu_isp_number
    global Edep_number, Edep_dq, Edep_x
    global Edep_z, Edep_trkID, nu_isp_pdg, Edep_y, Edep_t,Edep_t_end,Edep_length
    global check
    
    with uproot.open(HEAD_ROOT + FILE) as f:
        tree = f['event_tree']
        # print(tree.keys())
        Edep_x = tree.array(['hit_start_x'])
        g4_nParticles = tree.array(['number_particles'])
        g4_trkID = tree.array(['particle_track_id'])
        g4_PDG = tree.array(['particle_pdg_code'])
        Edep_number = tree.array(['number_hits'])
        Edep_dq = tree.array(['hit_energy_deposit'])
        Edep_y = tree.array(['hit_start_y'])
        Edep_x = tree.array(['hit_start_x'])
        Edep_z = tree.array(['hit_start_z'])
        
        
        
        Edep_t = tree.array(['hit_start_t'])
        Edep_trkID = tree.array(['hit_track_id'])
        Edep_t_end = tree.array(['hit_end_t'])
        Edep_length = tree.array(['hit_length'])
        # check = tree.array(['particle_daughter_track_id'])
    # print(check[0])

def getParticles(Event, PID):    
    b = np.array([g4_PDG[Event][np.in1d(g4_trkID[Event],Edep_trkID[Event][oo])][0] for oo in range(len(Edep_trkID[Event]))])

    return np.where(b == PID)[0].tolist()

def getP(Event):    
    b = np.array([g4_PDG[Event][np.in1d(g4_trkID[Event],Edep_trkID[Event][oo])][0] for oo in range(len(Edep_trkID[Event]))])
    aa = []
    for i in range(len(b)):
        if b[i] == b[i-1]:
            continue
        else:
            aa.append(b[i])
    aa = np.array(aa)
    return aa

def getD(Event, PID):    
    b = np.array([g4_PDG[Event][np.in1d(g4_trkID[Event],Edep_trkID[Event][oo])][0] for oo in range(len(Edep_trkID[Event]))])

    return np.where(b == PID)[0].tolist()

def getE(values):
    
    Energy = []
    for Edep in values:
        if((Edep_x[idx][Edep] + x_off) < 0 or (Edep_x[idx][Edep] + x_off) > xDim or
            (Edep_y[idx][Edep] + y_off) < 0 or (Edep_y[idx][Edep] + y_off) > yDim or
            (Edep_z[idx][Edep] + z_off) < 0 or (Edep_z[idx][Edep] + z_off) > zDim or
                Edep_t[idx][Edep] > 6000):
            continue
        Energy.append(Edep_dq[idx][Edep])
    
    return sum(np.array(Energy))

HEAD = 'C:\\Users/msroo/My Drive/UTA/Physics_UTA/PhD/Simulation/data/'
HEAD_ROOT = 'D:\\bangbang/'

getRoot('banger_6_31.root')

''' Detector Parameters'''
xDim = 350; yDim = 600; zDim = 230 
x_off = 0; y_off = 0; z_off = 0 

particles = [321,-13,14,-11,12,-14]

particle_names = ['k+','u+','nu_mu','e+','nu_e','nu_mu_bar']

idxs = [];E,II= [],[]; E2 = []; III = []
for oo in range(1000):
    idx = oo
    input = g4_PDG[idx].tolist()
    if ((input.count(particles[0]) == 1) & (input.count(particles[1]) ==1  ) & (input.count(particles[2]) ==1 ) &
        (input.count(particles[3]) ==1 ) & (input.count(particles[4]) ==1 )  & (input.count(particles[5]) == 1)):
        muons = getParticles(idx,-13)
        kaons = getParticles(idx,321)
        positrons = getParticles(idx,-11)
        k_energy = round(sum(Edep_dq[idx][kaons]),1)
        mu_energy = round(sum(Edep_dq[idx][muons]),1) 
        p_energy = round(sum(Edep_dq[idx][positrons]),1)
        if (k_energy > mu_energy) & (mu_energy > p_energy):
            zoo = np.arange(0,Edep_number[idx],1).tolist()
            E = getE(zoo)
            E2 = sum(Edep_dq[idx])
            if E == E2:
                II.append(idx)
            # else:
            #     III.append(idx)

idxs = np.array(II)
print(idxs.tolist())
# print(len(idxs))









