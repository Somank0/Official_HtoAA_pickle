import uproot
import math
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from time import time
import pickle
import tqdm
from torch_geometric.data import Data
import torch

# Modified by Somanko
import torch.nn.functional as F
import os
MISSING = -999999

########################################
# HGCAL Values                         #
########################################

'''
HGCAL_X_Min = -36
HGCAL_X_Max = 36

HGCAL_Y_Min = -36
HGCAL_Y_Max = 36

HGCAL_Z_Min = 13
HGCAL_Z_Max = 265

HGCAL_Min = 0
HGCAL_Max = 230
# HGCAL_Max_EE =3200
# HGCAL_Max_FH = 2200
# HGCAL_MAX_AH =800
'''

########################################
# ECAL Values                          #
########################################

X_Min = -150
X_Max = 150

Y_Min = -150
Y_Max = 150

EE_Z_Min = -330
EE_Z_Max = 330

EE_Eta_Min = -2.5
EE_Eta_Max = 2.5

#EB_Z_Min = -300
#EB_Z_Max = 300

EB_Eta_Min = -1.4
EB_Eta_Max = 1.4

Phi_Min = -np.pi
Phi_Max = np.pi

iEta_Min = -100
iEta_Max = 120

iPhi_Min = 1
iPhi_Max = 360

iX_Min = 1
iX_Max = 100

iY_Min = 1
iY_Max = 100

ECAL_Min = 0
ECAL_Max = 50

ES_Min =0
#ES_Max = 100
ES_Max =0.02

#############################################

def localfeat(eta,phi,energy,energy2=None):
	
	e = rescale(eta,Eta_Min,Eta_Max)
	p = rescale(phi,Phi_Min,Phi_Max)
	en = rescale(energy,ECAL_Min,ECAL_Max)
	#en2 = rescale(energy2,ECAL_Min,ECAL_Max)
	return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )

def detfeat(ieta,iphi,energy,energy2=None):
	
	return ak.concatenate(
            (
                ieta[:, :, None],
                iphi[:, :, None],
                energy[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )
	

def detfeat(ieta,iphi,energy):
	e = rescale(ieta,iEta_Min,iEta_Max)
	p = rescale(iphi,iPhi_Min,iPhi_Max)
	en = rescale(energy,ECAL_Min,ECAL_Max)
	return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
            ),
            -1,
        )

def localfeat(eta,phi,energy,det=None):
	
    e = rescale(eta,Eta_Min,Eta_Max)
    p = rescale(phi,Phi_Min,Phi_Max)
    en = rescale(energy,ECAL_Min,ECAL_Max)
	#en2 = rescale(energy2,ECAL_Min,ECAL_Max)
    if det is None :
     return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )
    else :
     return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
	 	det[:,:,None],
            ),
            -1,
        )

def detfeat(ieta,iphi,energy,det=None):
    if det is None :
        return ak.concatenate(
                (
                    ieta[:, :, None],
                    iphi[:, :, None],
                    energy[:, :, None],
        #	en2[:,:,None],
                ),
                -1,
           )
    else:
        return ak.concatenate(
                (
                    ieta[:, :, None],
                    iphi[:, :, None],
                    energy[:, :, None],
        	    det[:,:,None],
                ),
                -1,
            )
	
def setptetaphie(pt, eta, phi, e):
    pta = abs(pt)
    return np.array([np.cos(phi) * pta, np.sin(phi) * pt, np.sinh(eta) * pta, e])


def getMass(lvec):
    return np.sqrt(
        lvec[3] * lvec[3] - lvec[2] * lvec[2] - lvec[1] * lvec[1] - lvec[0] * lvec[0]
    )


def rescale(feature, minval, maxval):
    top = feature - minval
    bot = maxval - minval
    return top / bot


def dphi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    gt = dphi > np.pi
    dphi[gt] = 2 * np.pi - dphi[gt]
    return dphi


def dR(eta1, eta2, phi1, phi2):
    dp = dphi(phi1, phi2)
    de = np.abs(eta1 - eta2)

    return np.sqrt(dp * dp + de * de)


def cartfeat(x, y, z, E, det=None):
    #E = rescale(E, ECAL_Min, ECAL_Max)
    #x = rescale(x, X_Min, X_Max)
    #y = rescale(y, Y_Min, Y_Max)
    #z = rescale(z, Z_Min, Z_Max)


    if det is None :
        return ak.concatenate(
            (x[:, :, None], y[:, :, None], z[:, :, None], E[:, :, None]),
            -1,
        )
    else:
        return ak.concatenate(
            (
                x[:, :, None],
                y[:, :, None],
                z[:, :, None],
                E[:, :, None],
                det[:, :, None],
            ),
            -1,
        )


def torchify(feat, graph_x=None):
    data = [
        Data(x=torch.from_numpy(ak.to_numpy(Pho).astype(np.float32))) for Pho in feat
    ]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data


def npify(feat):
    t0 = time()
    data = [ak.to_numpy(Pho) for Pho in feat]
    print("took %f" % (time() - t0))
    return data


class Extract:
    def __init__(
        self,
        outfolder="pickles",
        path="merged.root",
        treeName="nTuplelize/T",
        path2=None,
    ):
        if path is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree = uproot.open("%s:%s" % (path, treeName))
        if path2 is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree2 = uproot.open("%s:%s" % (path2, treeName))

        self.outfolder = outfolder

    def read(self, N=None):
        varnames = [
            "Hit_ES_X_Pho1",
            "Hit_ES_Y_Pho1",
            "Hit_ES_Z_Pho1",
            "Hit_ES_Eta_Pho1",
            "Hit_ES_Phi_Pho1",
            "ES_RecHitEnPho1",
            "Hit_ES_Eta_Pho2",
            "Hit_ES_Phi_Pho2",
            "Hit_ES_X_Pho2" ,         
            "Hit_ES_Y_Pho2",
            "Hit_ES_Z_Pho2",
            "ES_RecHitEnPho2",
            "Hit_ES_Eta_Pho3",
            "Hit_ES_Phi_Pho3",
            "Hit_ES_X_Pho3",
            "Hit_ES_Y_Pho3",
            "Hit_ES_Z_Pho3",
            "ES_RecHitEnPho3",
            "Hit_ES_Eta_Pho4",
            "Hit_ES_Phi_Pho4",
            "Hit_ES_X_Pho4",
            "Hit_ES_Y_Pho4",
            "Hit_ES_Z_Pho4",
            "ES_RecHitEnPho4",
            "Hit_X_Pho1",
            "Hit_Y_Pho1",
            "Hit_Z_Pho1",
            "Hit_Eta_Pho1",
            "Hit_Phi_Pho1",
            "iEtaPho1",
            "iPhiPho1",
            "RecHitEnPho1",
            "RecHitFracPho1",
            "Hit_X_Pho2",
            "Hit_Y_Pho2",
            "Hit_Z_Pho2",
            "Hit_Eta_Pho2",
            "Hit_Phi_Pho2",
            "iEtaPho2",
            "iPhiPho2",
            "RecHitEnPho2",
            "RecHitFracPho2",
            "Hit_X_Pho3",
            "Hit_Y_Pho3",
            "Hit_Z_Pho3",
            "Hit_Eta_Pho3",
            "Hit_Phi_Pho3",
            "RecHitEnPho3",
            "RecHitFracPho3",
            "iEtaPho3",
            "iPhiPho3",
            "Hit_X_Pho4",
            "Hit_Y_Pho4",
            "Hit_Z_Pho4",
            "Hit_Eta_Pho4",
            "Hit_Phi_Pho4",
            "RecHitEnPho4",
            "RecHitFracPho4",
            "iEtaPho4",
            "iPhiPho4",
            "A_flags",
            "energy",
            "phi",
            "eta",
            "rho",
            "Pho_SigIEIE",
            "Pho_R9",
            "pt",
            "A_lead_Gen_mass",
            "A_lead_Gen_pt",
            "A_lead_Gen_eta",
            "A_lead_Gen_phi",
            "A_sublead_Gen_mass",
            "A_sublead_Gen_pt",
            "A_sublead_Gen_eta",
            "A_sublead_Gen_phi",
            "H_Gen_mass",
            "H_Gen_pt",
            "H_Gen_eta",
            "H_Gen_phi",
            "A_lead_Pho_Gen_Pt",
            "A_lead_Pho_Gen_Eta",
            "A_lead_Pho_Gen_Phi",
            "A_lead_Pho_Gen_E",
            "A_sublead_Pho_Gen_Pt",
            "A_sublead_Pho_Gen_Eta",
            "A_sublead_Pho_Gen_Phi",
            "A_sublead_Pho_Gen_E",
            "rho",
            "Pho_SigIEIE",
            "Pho_R9",

	    #"A_Gen_mass",
	    #"A_Gen_pt",
	    #"A_Gen_eta",
	    #"A_Gen_phi",
	    #"Pho_Gen_Pt",
        #"Pho_Gen_Eta",
        #"Pho_Gen_Phi",
	    #"pt"
        ]

        arrs = self.tree.arrays(varnames)
        arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho1"]]]
        # arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho2"]]]
        result = {}
        t0 = time()
        print("\tDumping target took %f seconds" % (time() - t0))
        print("Building cartesian features..")

        A1_mass = arrs["A_lead_Gen_mass"]
        A2_mass = arrs["A_sublead_Gen_mass"]
        A1_eta = arrs["A_lead_Gen_eta"]
        A2_eta = arrs["A_sublead_Gen_eta"]
        A1_eta = ak.flatten(A1_eta)
        A2_eta = ak.flatten(A2_eta)
        print(len(A1_mass))
        print(len(A2_mass))
        cond1_EB = abs(A1_eta) <=1.4
        cond1_EE = (abs(A1_eta)>=1.44) & (abs(A1_eta)<=2.5) 
        cond2_EB = abs(A2_eta) <=1.4
        cond2_EE = (abs(A2_eta)>=1.44) & (abs(A2_eta)<=2.5) 
        cond_EBEB = cond1_EB & cond2_EB
        cond_EBEE = (cond1_EB & cond2_EE)
        cond_EEEB = (cond1_EE & cond2_EB)
        cond_EEEE = (cond1_EE & cond2_EE)

        A1_mass_EBEB = A1_mass[ak.num(A1_mass[cond_EBEB])>0]
        A2_mass_EBEB = A2_mass[ak.num(A2_mass[cond_EBEB])>0]

        A1_mass_EBEE = A1_mass[ak.num(A1_mass[cond_EBEE])>0]
        A2_mass_EBEE = A2_mass[ak.num(A2_mass[cond_EBEE])>0]
        A1_mass_EEEB = A1_mass[ak.num(A1_mass[cond_EEEB])>0]
        A2_mass_EEEB = A1_mass[ak.num(A1_mass[cond_EEEB])>0]

        A1_mass_EEEE = A1_mass[ak.num(A1_mass[cond_EEEE])>0]
        A2_mass_EEEE = A2_mass[ak.num(A2_mass[cond_EEEE])>0]
        gen_alfg = A1_mass*0 +0
        gen_a2flg = A2_mass*0 +1
        #print(len(A1_mass))
        #print(len(A2_mass))
        print("EBEB ",len(A2_mass_EBEB))
        print("EBEE ",len(A2_mass_EBEE))
        print("EEEB ",len(A2_mass_EEEB))
        print("EEEE ",len(A2_mass_EEEE))

        target_EBEB = ak.concatenate((A1_mass_EBEB,A2_mass_EBEB),axis=0)
        target_EEEE = ak.concatenate((A1_mass_EEEE,A2_mass_EEEE),axis=0)
        mixed_target_EB = ak.concatenate((A1_mass_EBEE,A2_mass_EEEB),axis=0)
        mixed_target_EE = ak.concatenate((A1_mass_EEEB,A2_mass_EBEE),axis=0)

        ebeb_a1_flg =A1_mass_EBEB*0 + 0
        ebeb_a2_flg = A2_mass_EBEB*0 + 1
        ebeb_a_flg = ak.concatenate((ebeb_a1_flg,ebeb_a2_flg),axis=0)

        eeee_a1_flg =A1_mass_EEEE*0 + 0
        eeee_a2_flg = A2_mass_EEEE*0 + 1
        eeee_a_flg = ak.concatenate((eeee_a1_flg,eeee_a2_flg),axis=0)

        mixed_eb_a1_flg =A1_mass_EBEE*0 + 0
        mixed_eb_a2_flg = A2_mass_EEEB*0 + 1
        mixed_eb_a_flg = ak.concatenate((mixed_eb_a1_flg,mixed_eb_a2_flg),axis=0)

        mixed_ee_a1_flg = A1_mass_EEEB*0 + 0
        mixed_ee_a2_flg = A2_mass_EBEE*0 +1
        mixed_ee_a_flg = ak.concatenate((mixed_ee_a1_flg,mixed_ee_a2_flg),axis=0)

        os.makedirs("%s/EBEB"% (self.outfolder), exist_ok=True)
        os.makedirs("%s/EEEE"% (self.outfolder), exist_ok=True)
        os.makedirs("%s/EBEE/EB"% (self.outfolder), exist_ok=True)
        os.makedirs("%s/EBEE/EE"% (self.outfolder), exist_ok=True)

        with open("%s/EBEB/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(target_EBEB, outpickle)
        with open("%s/EBEB/a_flag.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(ebeb_a_flg, outpickle)
        with open("%s/EEEE/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(target_EEEE, outpickle)
        with open("%s/EEEE/a_flag.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(eeee_a_flg, outpickle)
        with open("%s/EBEE/EB/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(mixed_target_EB, outpickle)
        with open("%s/EBEE/EB/a_flag.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(mixed_eb_a_flg, outpickle)
        with open("%s/EBEE/EE/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(mixed_target_EE, outpickle)
        with open("%s/EBEE/EE/a_flag.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(mixed_ee_a_flg, outpickle)

        A_flags_o=arrs["A_flags"]
        A_flags=ak.pad_none(A_flags_o, 4)
        
        A_pho_hitx = ak.concatenate((arrs["Hit_X_Pho1"],arrs["Hit_X_Pho2"],arrs["Hit_X_Pho3"],arrs["Hit_X_Pho4"]),axis=1)
        A_pho_hity = ak.concatenate((arrs["Hit_Y_Pho1"],arrs["Hit_Y_Pho2"],arrs["Hit_Y_Pho3"],arrs["Hit_Y_Pho4"]),axis=1)
        A_pho_hitz = ak.concatenate((arrs["Hit_Z_Pho1"],arrs["Hit_Z_Pho2"],arrs["Hit_Z_Pho3"],arrs["Hit_Z_Pho4"]),axis=1)
        A_pho_hitE = ak.concatenate((arrs["RecHitEnPho1"],arrs["RecHitEnPho2"],arrs["RecHitEnPho3"],arrs["RecHitEnPho4"]),axis=1)
        A_pho_hitFrac = ak.concatenate((arrs["RecHitFracPho1"],arrs["RecHitFracPho2"],arrs["RecHitFracPho3"],arrs["RecHitFracPho4"]),axis=1)
        A_pho_hitE = rescale(A_pho_hitE,ECAL_Min,ECAL_Max)
       
        ES_hitx = ak.concatenate((arrs["Hit_ES_X_Pho1"],arrs["Hit_ES_X_Pho2"],arrs["Hit_ES_X_Pho3"],arrs["Hit_ES_X_Pho4"]),axis=1)
        ES_hity = ak.concatenate((arrs["Hit_ES_Y_Pho1"],arrs["Hit_ES_Y_Pho2"],arrs["Hit_ES_Y_Pho3"],arrs["Hit_ES_Y_Pho4"]),axis=1)
        ES_hitz = ak.concatenate((arrs["Hit_ES_Z_Pho1"],arrs["Hit_ES_Z_Pho2"],arrs["Hit_ES_Z_Pho3"],arrs["Hit_ES_Z_Pho4"]),axis=1)
        ES_hit_E = ak.concatenate((arrs["ES_RecHitEnPho1"],arrs["ES_RecHitEnPho2"],arrs["ES_RecHitEnPho3"],arrs["ES_RecHitEnPho4"]),axis=1)

        ES_hit_E = rescale(ES_hit_E,ES_Min,ES_Max)

        ES_flag = ES_hitx*0 +1
        EE_flag = A_pho_hitx*0 +0

        Flag = ak.concatenate((EE_flag,ES_flag),axis=-1)


        Hitx = ak.concatenate((A_pho_hitx,ES_hitx),axis=-1)
        Hity = ak.concatenate((A_pho_hity,ES_hity),axis=-1)
        Hitz = ak.concatenate((A_pho_hitz,ES_hitz),axis=-1)
        HitE = ak.concatenate((A_pho_hitE,ES_hit_E),axis=-1)
        Hitx = rescale(Hitx, X_Min,X_Max)
        Hity = rescale(Hity,Y_Min,Y_Max)
        Hitz = rescale(Hitz,EE_Z_Min,EE_Z_Max)

        A_flagPho1 = (ak.concatenate((arrs["Hit_X_Pho1"],arrs["Hit_ES_X_Pho1"]),axis=-1))*0 + A_flags[:,0][A_flags[:,0]>-1]
        A_flagPho2 = (ak.concatenate((arrs["Hit_X_Pho2"],arrs["Hit_ES_X_Pho2"]),axis=-1))*0 + A_flags[:,1][A_flags[:,1]>-1]
        A_flagPho3 = (ak.concatenate((arrs["Hit_X_Pho3"],arrs["Hit_ES_X_Pho3"]),axis=-1))*0 + A_flags[:,2][A_flags[:,2]>-1]
        A_flagPho4 = (ak.concatenate((arrs["Hit_X_Pho4"],arrs["Hit_ES_X_Pho4"]),axis=-1))*0 + A_flags[:,3][A_flags[:,3]>-1]
        A_Flag = ak.concatenate((A_flagPho1,A_flagPho2,A_flagPho3,A_flagPho4),axis=1)
        

        Pho1_flag = (ak.concatenate((arrs["Hit_X_Pho1"],arrs["Hit_ES_X_Pho1"]),axis=-1))*0 +0
        Pho2_flag = (ak.concatenate((arrs["Hit_X_Pho2"],arrs["Hit_ES_X_Pho2"]),axis=-1))*0 +1
        Pho3_flag = (ak.concatenate((arrs["Hit_X_Pho3"],arrs["Hit_ES_X_Pho3"]),axis=-1))*0 +2
        Pho4_flag = (ak.concatenate((arrs["Hit_X_Pho4"],arrs["Hit_ES_X_Pho4"]),axis=-1))*0 +3
        Pho_flag = ak.concatenate((Pho1_flag,Pho2_flag,Pho3_flag,Pho4_flag),axis=1)


        eta_o = arrs["eta"]
        eta = ak.pad_none(eta_o,4)
        eta_flags_pho1 = (ak.concatenate((arrs["Hit_X_Pho1"],arrs["Hit_ES_X_Pho1"]),axis=-1))*0 + eta[:,0][eta[:,0]>-999]
        eta_flags_pho2 = (ak.concatenate((arrs["Hit_X_Pho2"],arrs["Hit_ES_X_Pho2"]),axis=-1))*0 + eta[:,1][eta[:,1]>-999]
        eta_flags_pho3 = (ak.concatenate((arrs["Hit_X_Pho3"],arrs["Hit_ES_X_Pho3"]),axis=-1))*0 + eta[:,2][eta[:,2]>-999]
        eta_flags_pho4 = (ak.concatenate((arrs["Hit_X_Pho4"],arrs["Hit_ES_X_Pho4"]),axis=-1))*0 + eta[:,3][eta[:,3]>-999]
        eta_flags = ak.concatenate((eta_flags_pho1,eta_flags_pho2,eta_flags_pho3,eta_flags_pho4),axis=1)

        isEE = (abs(eta_flags)>=1.44) & (abs(eta_flags) <= 2.5)
        isEB = abs(eta_flags)<=1.4
        isA1 = A_Flag ==0
        isA2 = A_Flag ==1
        
        A1_hitx=Hitx[isA1]
        A1_hity=Hity[isA1]
        A1_hitz=Hitz[isA1]
        A1_hitE = HitE[isA1]
        A1_flag = Flag[isA1]
        A2_hitx=Hitx[isA2]
        A2_hity=Hity[isA2]
        A2_hitz=Hitz[isA2]
        A2_hitE = HitE[isA2]
        A2_flag = Flag[isA2]

        sig_IEIE = arrs["Pho_SigIEIE"]
        rho_o = arrs["rho"]
        A1_sc_eta = eta_o[A_flags_o==0]
        A1_pho_sigIEIE = sig_IEIE[A_flags_o==0]
        #print(A1_sc_eta)
        A2_sc_eta = eta_o[A_flags_o==1]
        A2_pho_sigIEIE = sig_IEIE[A_flags_o==1]

        t0=time()
        
        EBEB_sc_eta = ak.concatenate((A1_sc_eta[cond_EBEB],A2_sc_eta[cond_EBEB]),axis=0)
        EBEB_sc_eta = ak.drop_none(EBEB_sc_eta)
        EBEB_sc_eta = EBEB_sc_eta[ak.num(EBEB_sc_eta)>0]
        sc_eta1 =[]
        sc_eta2=[]
        for index,value in enumerate(EBEB_sc_eta):
            sc_eta1.append(EBEB_sc_eta[index][0])
            if len(value)==1:
                sc_eta2.append(0)
            if len(value)>=2:
                sc_eta2.append(EBEB_sc_eta[index][1])
        gx = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta1]
        gx2 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta2]
        with open("%s/EBEB/sc_eta1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx, outpickle)
        with open("%s/EBEB/sc_eta2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx2, outpickle)

        EBEB_graphx =[]
        EBEB_graphx.append(gx)
        EBEB_graphx.append(gx2)
        EBEB_graphx = np.concatenate(EBEB_graphx,1)

        EEEE_sc_eta = ak.concatenate((A1_sc_eta[cond_EEEE],A2_sc_eta[cond_EEEE]),axis=0)
        EEEE_sc_eta = ak.drop_none(EEEE_sc_eta)
        EEEE_sc_eta = EEEE_sc_eta[ak.num(EEEE_sc_eta)>0]
        EEEE_sigIEIE = ak.concatenate((A1_pho_sigIEIE[cond_EEEE],A2_pho_sigIEIE[cond_EEEE]),axis=0)
        EEEE_sigIEIE= ak.drop_none(EEEE_sigIEIE)
        EEEE_sigIEIE = EEEE_sigIEIE[ak.num(EEEE_sigIEIE)>0]
        EEEE_rho = rho_o[cond_EEEE]
        sc_eta1 =[]
        sc_eta2=[]
        sigIEIE1=[]
        sigIEIE2=[]
        for index,value in enumerate(EEEE_sc_eta):
            sc_eta1.append(EEEE_sc_eta[index][0])
            if len(value)==1:
                sc_eta2.append(0)
                sigIEIE2.append(0)
                sigIEIE1.append(EEEE_sigIEIE[index][0])
            if len(value)>=2:
                sc_eta2.append(EEEE_sc_eta[index][1])
                sigIEIE2.append(EEEE_sigIEIE[index][1])
        gx = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta1]
        gx2 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta2]
        gx3 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sigIEIE1]
        gx4 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sigIEIE2]
        gx5 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in EEEE_rho]
        with open("%s/EEEE/sc_eta1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx, outpickle)
        with open("%s/EEEE/sc_eta2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx2, outpickle)
        with open("%s/EEEE/sigIEiE1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx3, outpickle)
        with open("%s/EEEE/sigIEIE2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx4, outpickle)
        with open("%s/EEEE/Rho.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx5, outpickle)



        EEEE_graphx =[]
        EEEE_graphx.append(gx)
        EEEE_graphx.append(gx2)
        EEEE_graphx.append(gx3)
        EEEE_graphx.append(gx4)
        EEEE_graphx.append(gx5)
        EEEE_graphx = np.concatenate(EEEE_graphx,1)


        mixed_eb_sc_eta = ak.concatenate((A1_sc_eta[cond_EBEE],A2_sc_eta[cond_EEEB]),axis=0)
        mixed_eb_sc_eta = ak.drop_none(mixed_eb_sc_eta)
        mixed_eb_sc_eta = mixed_eb_sc_eta[ak.num(mixed_eb_sc_eta)>0]
        sc_eta1 =[]
        sc_eta2=[]
        for index,value in enumerate(mixed_eb_sc_eta):
            sc_eta1.append(value[0])
            if len(value)==1:
                sc_eta2.append(0)
            if len(value)>=2:
                sc_eta2.append(value[1])
        gx = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta1]
        gx2 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta2]
        with open("%s/EBEE/EB/sc_eta1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx, outpickle)
        with open("%s/EBEE/EB/sc_eta2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx2, outpickle)
        mixed_eb_graphx =[]
        mixed_eb_graphx.append(gx)
        mixed_eb_graphx.append(gx2)
        mixed_eb_graphx = np.concatenate(mixed_eb_graphx,1)

        mixed_ee_sc_eta = ak.concatenate((A1_sc_eta[cond_EEEB],A2_sc_eta[cond_EBEE]),axis=0)
        mixed_ee_sc_eta = ak.drop_none(mixed_ee_sc_eta)
        mixed_ee_sc_eta = mixed_ee_sc_eta[ak.num(mixed_ee_sc_eta)>0]
        sc_eta1 =[]
        sc_eta2=[]
        for index,value in enumerate(mixed_ee_sc_eta):
            sc_eta1.append(value[0])
            if len(value)==1:
                sc_eta2.append(0)
            if len(value)>=2:
                sc_eta2.append(value[1])
        gx = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta1]
        gx2 = [torch.from_numpy(np.asarray(ak.to_numpy(x)).astype(np.float32)) for x in sc_eta2]
        with open("%s/EBEE/EE/sc_eta1.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx, outpickle)
        with open("%s/EBEE/EE/sc_eta2.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(gx2, outpickle)
        mixed_ee_graphx =[]
        mixed_ee_graphx.append(gx)
        mixed_ee_graphx.append(gx2)
        mixed_ee_graphx = np.concatenate(mixed_ee_graphx,1)
        print("\tBuilding high level features took %f seconds" % (time() - t0))

        EBEB_hitx = ak.concatenate((A1_hitx[cond_EBEB],A2_hitx[cond_EBEB]),axis=0)
        EBEB_hity = ak.concatenate((A1_hity[cond_EBEB],A2_hity[cond_EBEB]),axis=0)
        EBEB_hitz = ak.concatenate((A1_hitz[cond_EBEB],A2_hitz[cond_EBEB]),axis=0)
        EBEB_hitE = ak.concatenate((A1_hitE[cond_EBEB],A2_hitE[cond_EBEB]),axis=0)
        EBEB_flags = ak.concatenate((A1_flag[cond_EBEB],A2_flag[cond_EBEB]),axis=0)
        EBEB_a_flag = ak.concatenate(((A_Flag[isA1])[cond_EBEB],(A_Flag[isA2])[cond_EBEB]),axis=0)

        mixed_EB_hitx = ak.concatenate((A1_hitx[cond_EBEE],A2_hitx[cond_EEEB]),axis=0)
        mixed_EB_hity = ak.concatenate((A1_hity[cond_EBEE],A2_hity[cond_EEEB]),axis=0)
        mixed_EB_hitz = ak.concatenate((A1_hitz[cond_EBEE],A2_hitz[cond_EEEB]),axis=0)
        mixed_EB_hitE = ak.concatenate((A1_hitE[cond_EBEE],A2_hitE[cond_EEEB]),axis=0)
        mixed_EB_flags = ak.concatenate((A1_flag[cond_EBEE],A2_flag[cond_EEEB]),axis=0)

        mixed_EE_hitx = ak.concatenate((A1_hitx[cond_EEEB],A2_hitx[cond_EBEE]),axis=0)
        mixed_EE_hity = ak.concatenate((A1_hity[cond_EEEB],A2_hity[cond_EBEE]),axis=0)
        mixed_EE_hitz = ak.concatenate((A1_hitz[cond_EEEB],A2_hitz[cond_EBEE]),axis=0)
        mixed_EE_hitE = ak.concatenate((A1_hitE[cond_EEEB],A2_hitE[cond_EBEE]),axis=0)
        mixed_EE_flags = ak.concatenate((A1_flag[cond_EEEB],A2_flag[cond_EBEE]),axis=0)


        EEEE_hitx = ak.concatenate((A1_hitx[cond_EEEE],A2_hitx[cond_EEEE]),axis=0)
        EEEE_hity = ak.concatenate((A1_hity[cond_EEEE],A2_hity[cond_EEEE]),axis=0)
        EEEE_hitz = ak.concatenate((A1_hitz[cond_EEEE],A2_hitz[cond_EEEE]),axis=0)
        EEEE_hitE = ak.concatenate((A1_hitE[cond_EEEE],A2_hitE[cond_EEEE]),axis=0)
        EEEE_flags = ak.concatenate((A1_flag[cond_EEEE],A2_flag[cond_EEEE]),axis=0)

        #print(EBEB_hitx)


        EBEB_cf = cartfeat(EBEB_hitx,EBEB_hity,EBEB_hitz,EBEB_hitE)
        #EBEB_cf = cartfeat(EBEB_hitx,EBEB_hity,EBEB_hitz,EBEB_hitE,EBEB_flags)
        EEEE_cf = cartfeat(EEEE_hitx,EEEE_hity,EEEE_hitz,EEEE_hitE,EEEE_flags)
        mixed_EB_cf = cartfeat(mixed_EB_hitx,mixed_EB_hity,mixed_EB_hitz,mixed_EB_hitE)
        #mixed_EB_cf = cartfeat(mixed_EB_hitx,mixed_EB_hity,mixed_EB_hitz,mixed_EB_hitE,mixed_EB_flags)
        mixed_EE_cf = cartfeat(mixed_EE_hitx,mixed_EE_hity,mixed_EE_hitz,mixed_EE_hitE,mixed_EE_flags)
        print("\tBuilding features took %f seconds" % (time() - t0))
        t0 = time()

        print(len(target_EBEB))
        print(len(EBEB_hitx))



        result["EBEB_cf"] = torchify(EBEB_cf)
        #result["EBEB_cf"] = torchify(EBEB_cf,EBEB_graphx)
        result["EEEE_cf"] = torchify(EEEE_cf,EEEE_graphx)
        result["mixed_EB_cf"] = torchify(mixed_EB_cf)
        #result["mixed_EB_cf"] = torchify(mixed_EB_cf,mixed_eb_graphx)
        result["mixed_EE_cf"] = torchify(mixed_EE_cf,mixed_ee_graphx)
        print("\tTorchifying took %f seconds" % (time() - t0))
        t0 = time()

        with open("%s/EBEB/cartfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["EBEB_cf"], f, pickle_protocol=4)
        with open("%s/EEEE/cartfeat_ES.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["EEEE_cf"], f, pickle_protocol=4)
        with open("%s/EBEE/cartfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["mixed_EB_cf"], f, pickle_protocol=4)
        with open("%s/EBEE/cartfeat_ES.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["mixed_EE_cf"], f, pickle_protocol=4)
        print("\tDumping took %f seconds" % (time() - t0))
        return result      
