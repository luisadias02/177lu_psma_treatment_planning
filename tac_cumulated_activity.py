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

    #kidney data
    mtp_data = pd.read_excel(data_path,sheet_name="lu177_mtp_data_kidneys",header=None,skiprows=2,usecols="A:G",
        names=["patient","right_t1", "right_t2", "right_t3","left_t1", "left_t2", "left_t3"])

    mtp_data = mtp_data.dropna(subset=["patient"])
    mtp_list = [{
            "patient": row["patient"],
            "right_kidney": [row["right_t1"], row["right_t2"], row["right_t3"]],
            "left_kidney": [row["left_t1"], row["left_t2"], row["left_t3"]]}
        for _, row in mtp_data.iterrows()]
    #print(mtp_list)

    #liver, spleen and bkg data
    mtp_data_ot = pd.read_excel(data_path,sheet_name="lu177_mtp_data_others",header=None,skiprows=2,usecols="A:J",
        names=["patient","liver_t1", "liver_t2", "liver_t3","spleen_t1", "spleen_t2", "spleen_t3", "bkg_t1", "bkg_t2", "bkg_t3"])

    mtp_data_ot = mtp_data_ot.dropna(subset=["patient"])
    mtp_list_ot = [{
            "patient": row["patient"],
            "liver": [row["liver_t1"], row["liver_t2"], row["liver_t3"]],
            "spleen": [row["spleen_t1"], row["spleen_t2"], row["spleen_t3"]],
            "bkg":[row["bkg_t1"], row["bkg_t2"], row["bkg_t3"]] }
        for _, row in mtp_data_ot.iterrows()]
    #print(mtp_list_ot)

    #tumours data
    tumours_data = pd.read_excel(data_path, sheet_name="lu177_mtp_data_tumours", header=None, skiprows=2)  
    tumours_data = tumours_data.dropna(subset=[0])          # column 0 = patient
    tumours_list = []
    for _, row in tumours_data.iterrows():
        vals = row.iloc[1:].tolist()
        tumour_vals = [vals[i:i+3] for i in range(0, len(vals), 3)]          # [t1, t2, t3] per tumour
        tumour_vals = [t for t in tumour_vals if not all(pd.isna(x) for x in t)]  
        tumours_list.append({"patient": row[0], "tumours": tumour_vals})
    #print(tumours_list)

    #SPECT time-points
    time_points_data = pd.read_excel(data_path,sheet_name="lu177_time_points",header=None,skiprows=1,usecols="A:D",
        names=["patient","t1", "t2", "t3"])

    time_points_data = time_points_data.dropna(subset=["patient"])
    time_points_list = [{
            "patient": row["patient"],
            "time_points": [row["t1"], row["t2"], row["t3"]]}
        for _, row in time_points_data.iterrows()]
    #print(time_points_list)

    #measured activities
    activities_data = pd.read_excel(data_path,sheet_name="activities_100i_25",header=None,skiprows=1,usecols="A:I",
        names=["patient","A_lu_admin", "A_24h_lu_rk","A_24h_lu_lk", "A_24h_ac_rk", "A_24h_ac_lk", "A_24h_lu_liver",
               "A_24h_lu_spleen", "A_24h_lu_bkg"])

    activities_data = activities_data.dropna(subset=["patient"])
    activities_list = [{
            "patient": row["patient"],
            "activities": [row["A_lu_admin"], row["A_24h_lu_rk"], row["A_24h_lu_lk"], row["A_24h_ac_rk"], row["A_24h_ac_lk"], 
                           row["A_24h_lu_liver"], row["A_24h_lu_spleen"], row["A_24h_lu_bkg"] ]}
        for _, row in activities_data.iterrows()]
    #print(activities_list)

    #tumours measured data (24h)
    sim_data = pd.read_excel(data_path, sheet_name="activities_tumours_lu177", header=None, skiprows=2, usecols="A:O")  # sheet name: adjust
    sim_data = sim_data.dropna(subset=[0])          
    sim_list_tumours = []
    for _, row in sim_data.iterrows():
        vals = row.iloc[1:].tolist()
        while vals and pd.isna(vals[-1]):           
            vals.pop()
        sim_list_tumours.append({"patient": row[0], "simulated_tumours_24h": vals})
    #print(sim_list_tumours)


    return mtp_list, mtp_list_ot, time_points_list, activities_list, tumours_list, sim_list_tumours


def tac_fitting(activity_values:list, time_points:list, organ:str, plot:bool):

    def mono_exponential_fit(t0, t1, t2, y0, y1, y2, k1_init, k2_init, c1_init,k):

        t=[t0, t1, t2]
        y=[y0, y1, y2]
        if y2==None:
            t=[t0, t1]
            y=[y0, y1]

        def mono_exp(t, c1, k1):
            return (c1*np.exp(-(k1+k)*t))
        
        mono_model= Model(mono_exp)       
        params = mono_model.make_params(k1=k1_init,c1=c1_init) 
        params['k1'].min = 0 
        result= mono_model.fit(y, params, method='leastsq', t=t)
        return result

    def bi_exponential_fit(t0, t1, t2, y0, y1, y2, k1_init, k2_init, c1_init,k):
        t=[t0, t1, t2]
        y=[y0, y1, y2]

        def bi_exp(t, c1, k1, k2):
            return (-c1*np.exp(-(k2+k)*t) + c1*np.exp(-(k1+k)*t))
        
        bi_model= Model(bi_exp)       
        params = bi_model.make_params(k1=k1_init,c1=c1_init, k2 =k2_init) 
        params['k1'].min = 0 
        result= bi_model.fit(y, params, method='leastsq', t=t)
        return result

    time_points = [x for x in time_points if str(x) != 'nan']
    activity_values = [x for x in activity_values if str(x) != 'nan']

    if len(time_points)==2:
        t0, t1= time_points[0], time_points[1]
        t2=None
        y0, y1= activity_values[0], activity_values[1]
    else:    
        t0, t1, t2= time_points[0], time_points[1], time_points[2]
        y0, y1, y2 = activity_values[0], activity_values[1], activity_values[2]

    if (y1 < y0) and (len(time_points)>2):
        #print("monoexponential fit")

        y0, y1, y2 = activity_values[0], activity_values[1], activity_values[2]
        k1_init, k2_init, c1_init = 0.1, 0.005, activity_values[0]
        fitted_curve = mono_exponential_fit(t0, t1, t2, y0, y1, y2,k1_init, k2_init, c1_init, k_phys_lu)
        params = ['c1', 'k1']
        fitted_parameters = [fitted_curve.params[p].value for p in params]

    elif len(time_points)==2:
        y0, y1 = activity_values[0], activity_values[1]
        k1_init, k2_init, c1_init = 0.1, 0.005, activity_values[0]
        fitted_curve = mono_exponential_fit(t0, t1, t2, y0, y1, y2=None, k1_init= k1_init, k2_init= k2_init, c1_init= c1_init, k= k_phys_lu)
        params = ['c1', 'k1']
        fitted_parameters = [fitted_curve.params[p].value for p in params]

    else:
        #print("biexponential fit")
        k1_init, k2_init, c1_init = 0.1, 0.005, activity_values[0]
        fitted_curve = bi_exponential_fit(t0, t1, t2, y0, y1, y2,k1_init, k2_init, c1_init, k_phys_lu)
        params = ['c1', 'k1', 'k2']
        fitted_parameters = [fitted_curve.params[p].value for p in params]

    if plot ==True:
        times = np.array([t0, t1, t2])
        activities = np.array([y0, y1, y2]) 

        t_fine = np.linspace(0, 210, 300)
        y_fitted = fitted_curve.eval(t=t_fine) 

        plt.plot(t_fine, y_fitted, linewidth=2.5, linestyle='--', label={organ})
        plt.scatter(times, activities, s=70, zorder=3)

        plt.legend()
        plt.tight_layout()
        plt.show()

    return fitted_parameters


def tac_fitting_lesions(activity_values:list, time_points:list, organ:str, plot:bool):

    def mono_exponential_fit(t0, t1, t2, y0, y1, y2, k1_init, k2_init, c1_init,k):

        t=[t0, t1, t2]
        y=[y0, y1, y2]
        if y2==None:
            t=[t0, t1]
            y=[y0, y1]

        def mono_exp(t, c1, k1):
            return (c1*np.exp(-(k1+k)*t))
        
        mono_model= Model(mono_exp)       
        params = mono_model.make_params(k1=k1_init,c1=c1_init) 
        params['k1'].min = 0 
        result= mono_model.fit(y, params, method='leastsq', t=t)
        return result

    def bi_exponential_fit(t0, t1, t2, y0, y1, y2, k1_init, k2_init, c1_init,k):
        t=[t0, t1, t2]
        y=[y0, y1, y2]

        def bi_exp(t, c1, k1, k2):
            return (-c1*np.exp(-(k2+k)*t) + c1*np.exp(-(k1+k)*t))
        
        bi_model= Model(bi_exp)
        params = bi_model.make_params(c1=-1.1*max(y), k1=k1_init, k2=k2_init)
        params['c1'].max = 0                                  # c1 <= 0 (k1 > k2)
        params['k2'].min = 0                                  # washout >= 0 (Teff <= físico)
        params.add('delta', value=max(k1_init - k2_init, 1e-3), min=1e-6, max=10)
        params['k1'].set(expr='k2 + delta')                   # k1 (captação) > k2 (washout)
        result= bi_model.fit(y, params, method='leastsq', t=t)
        return result

    time_points = [x for x in time_points if str(x) != 'nan']
    activity_values = [x for x in activity_values if str(x) != 'nan']

    if len(time_points)==2:
        t0, t1= time_points[0], time_points[1]
        t2=None
        y0, y1= activity_values[0], activity_values[1]
    else:    
        t0, t1, t2= time_points[0], time_points[1], time_points[2]
        y0, y1, y2 = activity_values[0], activity_values[1], activity_values[2]

    if (y1 < y0) and (len(time_points)>2):
        #print("monoexponential fit")

        y0, y1, y2 = activity_values[0], activity_values[1], activity_values[2]
        k1_init, k2_init, c1_init = 0.1, 0.005, activity_values[0]
        fitted_curve = mono_exponential_fit(t0, t1, t2, y0, y1, y2,k1_init, k2_init, c1_init, k_phys_lu)
        params = ['c1', 'k1']
        fitted_parameters = [fitted_curve.params[p].value for p in params]

    elif len(time_points)==2:
        y0, y1 = activity_values[0], activity_values[1]
        k1_init, k2_init, c1_init = 0.1, 0.005, activity_values[0]
        fitted_curve = mono_exponential_fit(t0, t1, t2, y0, y1, y2=None, k1_init= k1_init, k2_init= k2_init, c1_init= c1_init, k= k_phys_lu)
        params = ['c1', 'k1']
        fitted_parameters = [fitted_curve.params[p].value for p in params]

    else:
        #print("biexponential fit")
        k1_init, k2_init, c1_init = 0.1, 0.005, activity_values[0]
        fitted_curve = bi_exponential_fit(t0, t1, t2, y0, y1, y2,k1_init, k2_init, c1_init, k_phys_lu)
        params = ['c1', 'k1', 'k2']
        fitted_parameters = [fitted_curve.params[p].value for p in params]

    if plot ==True:
        times = np.array([t0, t1, t2])
        activities = np.array([y0, y1, y2]) 

        t_fine = np.linspace(0, 210, 300)
        y_fitted = fitted_curve.eval(t=t_fine) 

        plt.plot(t_fine, y_fitted, linewidth=2.5, linestyle='--', label={organ})
        plt.scatter(times, activities, s=70, zorder=3)

        plt.legend()
        plt.tight_layout()
        plt.show()

    return fitted_parameters



def time_integration(params:list, k_phys:float):
    """
    returns TIA in MBq*h
    """
    if len(params)==2:
        c1, k1= params[0], params[1]
        tia= c1/(k1+k_phys)
    else:
        c1, k1, k2= params[0], params[1], params[2]
        tia= -c1/(k2+k_phys) + c1/(k1+k_phys)

    return tia


def main(patient_id, true_dosimetry=False):

    data_path = 'patients_kidneys.xlsx'
    mtp_list, mtp_list_ot, time_points_list, activities_list, tumours_list, tumours_activities_list= read_data(data_path)

    patient= mtp_list[patient_id]["patient"] 
    print(f"patient {patient}")
    rk_quant= mtp_list[patient_id]["right_kidney"]  # right kidney quantification
    lk_quant= mtp_list[patient_id]["left_kidney"]   # left kidney quantification
    time_points= time_points_list[patient_id]["time_points"] #SPECT acquisition time points
    activities= activities_list[patient_id]["activities"] #activities post simulation at 24h
    tumours_quant= tumours_list[patient_id]["tumours"] # tumour quantification
    activities_tumours= tumours_activities_list[patient_id]["simulated_tumours_24h"] #activities post simulation at 24h

    #for Lu-177
    liver_quant= mtp_list_ot[patient_id]["liver"] # liver quantification
    spleen_quant= mtp_list_ot[patient_id]["spleen"] # spleen quantification
    bkg_quant= mtp_list_ot[patient_id]["bkg"] # background quantification

    rk_params= tac_fitting(rk_quant, time_points, "right kidney", plot=False) #biological half-life parameters not corrected for IA
    lk_params= tac_fitting(lk_quant, time_points, "left kidney", plot=False) #biological half-life parameters not corrected for IA

    liver_params=tac_fitting(liver_quant, time_points, "liver", plot=False)
    spleen_params= tac_fitting(spleen_quant, time_points, "spleen", plot=False)
    bkg_params= tac_fitting(bkg_quant, time_points, "background", plot=False)

    scale_factor_lu_rk= activities[1]/rk_quant[1]
    scale_factor_lu_lk= activities[2]/lk_quant[1]
    scale_factor_liver= activities[5]/liver_quant[1]
    scale_factor_spleen= activities[6]/spleen_quant[1]
    scale_factor_bkg= activities[7]/bkg_quant[1]

    if true_dosimetry==True:
        a_admin= activities[0] #Lu-177 original administered activity
        scale_factor= 7400/a_admin 
        print(f"administered activity: {a_admin} MBq, scale factor = {scale_factor :.3f}")
        scale_factor_lu_rk= scale_factor_lu_lk = scale_factor_liver =  scale_factor_spleen = scale_factor_bkg = scale_factor

    tumour_params = []
    scale_factors_tumours = []
    for i, (quant, act) in enumerate(zip(tumours_quant, activities_tumours)):
        params_init = tac_fitting_lesions(quant, time_points, f"tumour {i+1}", plot=False)   # not corrected for IA

        def extract_half_life(params):
            if len(params) == 2:
                c1, k1 = params
                k_eff = k1 + k_phys_lu
                if k_eff > 0:
                    params = params
                else:
                    c1=c1
                    k1=0
                    params= c1, k1
            elif len(params) == 3:
                c1, k1, k2 = params
                k_eff = k2 + k_phys_lu
                if k_eff > 0:
                    params = params
                else:
                    c1=c1
                    k2=0
                    params= c1, k1
            return params            

        if true_dosimetry == True:
            sf = 7400 / activities[0]
        elif pd.isna(act) or pd.isna(quant[1]) or quant[1] == 0:
            sf = np.nan
        else:
            sf = act / quant[1]
        
        params= extract_half_life(params_init)
        tumour_params.append(params)
        scale_factors_tumours.append(sf)

    #Because the DVK is in mGy/(MBq*s):
    tia_rk_lu= time_integration(rk_params, k_phys_lu)*scale_factor_lu_rk *3600  #Lu-177 scaled TIA in MBq*s: right kidney
    tia_lk_lu= time_integration(lk_params, k_phys_lu)*scale_factor_lu_lk *3600  #Lu-177 scaled TIA in MBq*s: left kidney

    tia_liver= time_integration(liver_params, k_phys_lu)*scale_factor_liver *3600 #Lu-177 scaled TIA in MBq*s: liver
    tia_spleen= time_integration(spleen_params, k_phys_lu)*scale_factor_spleen *3600 #Lu-177 scaled TIA in MBq*s: spleen
    tia_bkg= time_integration(bkg_params, k_phys_lu)*scale_factor_bkg *3600 #Lu-177 scaled TIA in MBq*s: background

    tia_tumours_lu = []
    for params, sf in zip(tumour_params, scale_factors_tumours):
        tia_tumours_lu.append(time_integration(params, k_phys_lu) * sf * 3600)   # Lu-177 scaled TIA in MBq*s: one per tumour

    return tia_rk_lu, tia_lk_lu, tia_liver, tia_spleen, tia_bkg, tia_tumours_lu
