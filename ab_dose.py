import sys
import os
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
import SimpleITK as sitk
from SimpleITK import ResampleImageFilter, Transform
import matplotlib.pyplot as plt
sys.path.append('C:\\ac225\\Ac-225')
#sys.path.append('/content/drive/MyDrive')
import functions

k_phys_lu= np.log(2)/(6.67*24) #Lu-177 physical decay constant (h-1)
k_phys_ac= np.log(2)/(9.92*24) #Ac-225 physical decay constant (h-1)


def read_data(data_path:str):

    mass_data = pd.read_excel(data_path,sheet_name="organ_mass",header=None,skiprows=1,usecols="A:C",
        names=["patient","rk", "lk"])

    mass_data = mass_data.dropna(subset=["patient"])
    mass_list = [{
            "patient": row["patient"],
            "rk_mass": [row["rk"]],
            "lk_mass": [row["lk"]]}
        for _, row in mass_data.iterrows()]
    #print(mtp_list)
    return mass_list

def abdose( masses:list, tia):
    e = 1.602E-19
    mass_rk, mass_lk=masses[0], masses[1]
    tia_rk_ac, tia_lk_ac= tia[0], tia[1]

    ## Mean energy of alpha emissions (eV) - The square brackets are the alpha energy values from ICRP Pub 107.
    Ac_alpha_mean = 5787.126283E+03  # Actinium  [5.892 MeV]
    Fr_alpha_mean = 6303.725230E+03  # Francium  [6.4199 MeV]
    At_alpha_mean = 7066.628656E+03  # Astatine  [7.2008 MeV]
    Bi_alpha_mean = 5848.188318E+03  # Bismuth [0.1245 MeV]
    Po_alpha_mean = 8376.000E+03  # Polonium [8.5370]

    # Determine S-values for alpha emissions
    Ac_sKid_rk = (Ac_alpha_mean*e) / mass_rk
    Fr_sKid_rk = (Fr_alpha_mean*e) / mass_rk
    At_sKid_rk = (At_alpha_mean*e) / mass_rk 
    Bi_sKid_rk = (Bi_alpha_mean*e) / mass_rk
    Po_sKid_rk = (Po_alpha_mean*e) / mass_rk

    Ac_sKid_lk = (Ac_alpha_mean*e) / mass_lk
    Fr_sKid_lk = (Fr_alpha_mean*e) / mass_lk
    At_sKid_lk = (At_alpha_mean*e) / mass_lk 
    Bi_sKid_lk = (Bi_alpha_mean*e) / mass_lk
    Po_sKid_lk = (Po_alpha_mean*e) / mass_lk

    # Calculate Dose (Gy)
    Ac_Dose_rk = tia_rk_ac*Ac_sKid_rk
    Fr_Dose_rk = tia_rk_ac*Fr_sKid_rk
    At_Dose_rk = tia_rk_ac*At_sKid_rk
    Bi_Dose_rk = 0.0216*tia_rk_ac*Bi_sKid_rk # Branching ratio included
    Po_Dose_rk = 0.9784*tia_rk_ac*Po_sKid_rk # Branching ratio included
    Tot_Dose_rk = Ac_Dose_rk+Fr_Dose_rk+At_Dose_rk+Bi_Dose_rk+Po_Dose_rk

    Ac_Dose_lk = tia_lk_ac*Ac_sKid_lk
    Fr_Dose_lk = tia_lk_ac*Fr_sKid_lk
    At_Dose_lk = tia_lk_ac*At_sKid_lk
    Bi_Dose_lk = 0.0216*tia_lk_ac*Bi_sKid_lk # Branching ratio included
    Po_Dose_lk = 0.9784*tia_lk_ac*Po_sKid_lk # Branching ratio included
    Tot_Dose_lk = Ac_Dose_lk+Fr_Dose_lk+At_Dose_lk+Bi_Dose_lk+Po_Dose_lk

    return [Tot_Dose_rk, Tot_Dose_lk] 

def main(patient_id:int, tia_values:list):

    data_path = 'patients_kidneys.xlsx'
    organ_mass_list = read_data(data_path)

    rk_mass= organ_mass_list[patient_id]["rk_mass"][0]
    lk_mass= organ_mass_list[patient_id]["lk_mass"][0]
    masses=[rk_mass, lk_mass]

    tia=[tia_values[2], tia_values[3]]
    tot_dose= abdose(masses, tia) #in Gy

    return tot_dose












