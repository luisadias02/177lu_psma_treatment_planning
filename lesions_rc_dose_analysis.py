import sys
import os
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
import SimpleITK as sitk
from SimpleITK import ResampleImageFilter, Transform
from scipy import signal
import matplotlib.pyplot as plt
sys.path.append('C:\\ac225\\Ac-225')
#sys.path.append('/content/drive/MyDrive')
import functions

k_phys_lu= np.log(2)/(6.67*24) #Lu-177 physical decay constant (h-1)

### LESION ANALYSIS

def read_data(data_path:str, tac:False):
    tumours_data = pd.read_excel(data_path, sheet_name="lu177_mtp_data_tumours", header=None, skiprows=2)  
    tumours_data = tumours_data.dropna(subset=[0])          # column 0 = patient
    tumours_list = []
    for _, row in tumours_data.iterrows():
        vals = row.iloc[1:].tolist()
        tumour_vals = [vals[i:i+3] for i in range(0, len(vals), 3)]          # [t1, t2, t3] per tumour
        tumour_vals = [t for t in tumour_vals if not all(pd.isna(x) for x in t)]  
        tumours_list.append({"patient": row[0], "tumours": tumour_vals})

    ia_data = pd.read_excel(data_path,sheet_name="activities_100i_25",header=None,skiprows=1,usecols="A:B",
        names=["patient","A_lu_admin"])
    ia_data = ia_data.dropna(subset=["patient"])
    ia_list = [{
            "patient": row["patient"],
            "activities": [row["A_lu_admin"]]}
        for _, row in ia_data.iterrows()]

    if tac == False:
        return tumours_list, ia_list
    
    else:
        time_points_data = pd.read_excel(data_path,sheet_name="lu177_time_points",header=None,skiprows=1,usecols="A:D",
            names=["patient","t1", "t2", "t3"])

        time_points_data = time_points_data.dropna(subset=["patient"])
        time_points_list = [{
                "patient": row["patient"],
                "time_points": [row["t1"], row["t2"], row["t3"]]}
            for _, row in time_points_data.iterrows()]
        #print(time_points_list)
        return tumours_list, ia_list, time_points_list

def quantification(volume: np.array, seg: np.array, key:int):
    mask = seg == key
    nvox = mask.sum()
    mean = volume[mask].mean()
    total= mean * nvox
    return total

def quantification_mean(volume: np.array, seg: np.array, key:int):
    mask = seg == key
    mean = volume[mask].mean()
    return mean

def lesion_volume(seg: np.array, key:int):
    mask = seg == key
    nvox = mask.sum()
    return nvox

def read_resample(seg_path:str, recon_path:str):
    def read_dicom(path):
        image= functions.read_dicom(path)
        return image
    image_recon= read_dicom(recon_path)
    image_seg= read_dicom(seg_path)  

    def resample_seg(image_recon, image_seg):
        resampler = ResampleImageFilter()
        resampler.SetReferenceImage(image_recon)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(Transform())
        resampled_seg_img = resampler.Execute(image_seg)
        return resampled_seg_img 
        
    resampled_seg= resample_seg(image_recon, image_seg) 
    return sitk.GetArrayFromImage(resampled_seg)

def lesion_selection(seg_path:str, tumours_keys:list, abdose_map:str):

    abdose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_map))

    seg_img= functions.read_dicom(seg_path)
    seg_arr= sitk.GetArrayFromImage(seg_img)
    voxel_volume= np.prod(seg_img.GetSpacing())*(1e-3)
    resampled_seg= read_resample(seg_path, abdose_map)

    init_dose=1000
    lesion_id=0

    for key in tumours_keys:
        nvox= lesion_volume(seg_arr, key)
        vol= nvox * voxel_volume
        if vol > 3:
            mean_dose_init= (quantification_mean(abdose_arr, resampled_seg, key))*7.4
            if mean_dose_init < init_dose and mean_dose_init > 0.1:
                init_uptake= mean_dose_init
                lesion_id= key

    return lesion_id

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
        if len(time_points) ==2:
            times = np.array([t0, t1])
            activities = np.array([y0, y1]) 
        else:
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

def extract_half_life(params, k_phys_lu):
    if len(params) == 2:
        c1, k1 = params
        k_eff = k1 + k_phys_lu
        return np.log(2) / k_eff if k_eff > 0 else np.nan

    elif len(params) == 3:
        c1, k1, k2 = params
        k_eff = k2 + k_phys_lu
        return np.log(2) / k_eff if k_eff > 0 else np.nan

def half_lives(patients:list, data_path:str, tumours_keys:dict):

    tumours_list, ia_list, time_points_data= read_data(data_path, tac=True)
    all_values={}
    for i in range(len(patients)):
        list_patient= tumours_list[i]
        patient= list_patient['patient'] #patient id
        list_patient_time_points=time_points_data[i]
        tumours_values_keys= tumours_keys[patient] #tumour keys
        tumours_activities = list_patient['tumours'] #tumours activities in the 3 time points
        time_points= list_patient_time_points['time_points'] #patient spect time-points

        spect_24h_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_lu177.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"

        seg_array= sitk.GetArrayFromImage(functions.read_dicom(seg_path))
        spect_arr= sitk.GetArrayFromImage(functions.read_dicom(spect_24h_path))
        seg_voxel_vol = np.prod(functions.read_dicom(seg_path).GetSpacing())*1e-3 #mL
        patient_dict= {}
        for j in range(len(tumours_values_keys)):
            tumour_key= tumours_values_keys[j] 
            lesion_vol= lesion_volume(seg_array, tumour_key)*seg_voxel_vol
            act= quantification(spect_arr, seg_array, tumour_key) #24h quantification
            lesion_name=f"{patient}_{tumour_key}"
            if act > 0.01 and lesion_vol >3 and lesion_name != 'p17_25': #excludes lesions outside the FOV and below 3mL
                quant= tumours_activities[j]
                params= tac_fitting_lesions(quant, time_points, f"tumour {tumour_key}", plot= False)   # not corrected for IA    
                half_life= extract_half_life(params, k_phys_lu)
                patient_dict[lesion_name]= half_life
                all_values[lesion_name]= half_life
            else:
                print(f"excluded lesion: {lesion_name}")

        teff_values= [value for value in patient_dict.values()]
        print(f"patient {patient} mean Teff (h) = {np.mean(teff_values):.2f} ({np.min(teff_values):.2f}, {np.max(teff_values):.2f}) ")

    teff_values= [value for value in all_values.values()]
    print(f"mean (all patients) Teff= {np.mean(teff_values):.2f} h +/- {np.std(teff_values):.2f}")

def rc_lesions(patients:list, data_path:str, tumours_keys:dict):

    tumours_list, ia_list= read_data(data_path, tac=False)
    all_lesions_rc= {}
    selected_lesions_rc= {} #dict for the selected lesion in each patient
    for i in range(len(patients)):
        list_patient= tumours_list[i]
        patient= list_patient['patient']
        print(f"reading patient {patient}")
        ia_patient= ia_list[i]['activities'][0]
        tumours_activities = list_patient['tumours']
        tumours_values= tumours_keys[patient]
        spect_24h_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_lu177.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"
        abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21.nii"
        patient_lesion_id= lesion_selection(seg_path, tumours_values, abdose_path) #id for the selected lesion in the optimisation algorithm

        spect_arr= sitk.GetArrayFromImage(functions.read_dicom(spect_24h_path))
        seg_array= sitk.GetArrayFromImage(functions.read_dicom(seg_path))
        seg_voxel_vol = np.prod(functions.read_dicom(seg_path).GetSpacing())*1e-3 #mL

        for j in range(len(tumours_values)):
            true_act= tumours_activities[j][1]/ia_patient * 7400 #24h true quantification at 7.4 GBq
            tumour_key = tumours_values[j]
            lesion_name=f"{patient}_{tumour_key}"
            act= quantification(spect_arr, seg_array, tumour_key) #24h quantification
            if act > 0.01 and lesion_name != 'p17_25': #excludes lesions outside the FOV
                rc= (act/true_act)*100
                lesion_vol= lesion_volume(seg_array, tumour_key)*seg_voxel_vol
                if lesion_vol > 3: #excludes lesions with volume < 3 mL
                    all_lesions_rc[lesion_name] = [rc, lesion_vol]
                    if tumour_key == patient_lesion_id:
                        selected_lesions_rc[lesion_name] = [rc, lesion_vol]
                        print(f"(selected lesion) tumour key {tumour_key}: true value = {true_act :.2f}, act= {act :.2f}, rc= {rc :.2f}%")
                    else:
                        print(f"tumour key {tumour_key}: true value = {true_act :.2f}, act= {act :.2f}, rc= {rc :.2f}%")
                else:
                    print(f"tumour key {tumour_key} less than 3 mL")
            else:
                print(f"tumour key {tumour_key} excluded")

    rc_values = [values[0] for values in all_lesions_rc.values()]
    print(f"total lesions evaluated: {len(rc_values)}")
    print(f"mean RC across evaluated lesions: {np.mean(rc_values) :.2f}%, ({np.min(rc_values) :.2f}, {np.max(rc_values) :.2f})")

    volumes = [values[1] for values in all_lesions_rc.values()]

    rc_selected= [values_1[0] for values_1 in selected_lesions_rc.values()]
    print(f"median RC across selected lesions: {np.median(rc_selected) :.2f}%, range({np.min(rc_selected) :.2f}, {np.max(rc_selected) :.2f})%")
    print(f"{sum(i>100 for i in rc_values)} lesions above 100\%")
    volumes_selected = [values_1[1] for values_1 in selected_lesions_rc.values()]

    plt.figure(figsize=(8, 6))
    plt.scatter(volumes, rc_values)
    plt.scatter(volumes_selected, rc_selected, c='g')
    plt.axhline(y=100, xmin=0, xmax= np.max(volumes), ls='--')
    plt.xlabel("Lesion volume (mL)")
    plt.ylabel("Recovery coefficient (%)")
    plt.title("Recovery coefficient vs lesion volume")
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.tight_layout()
    plt.show()

def abdose_lesions(patients:list, tumours_keys:dict, true_dosimetry:bool):

    for i in range(len(patients)):
        patient= patients[i]
        tumours_values= tumours_keys[patient]

        patient_abdoses={}
        patient_selected_lesion={}

        spect_24h_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_lu177.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"

        if true_dosimetry == False:
            abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21.nii"
        else:
            abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        
        patient_lesion_id= lesion_selection(seg_path, tumours_values, abdose_path) #id for the selected lesion in the optimisation algorithm
        
        spect_arr= sitk.GetArrayFromImage(functions.read_dicom(spect_24h_path))
        abdose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path)) #in Gy/GBq
        seg_array= sitk.GetArrayFromImage(functions.read_dicom(seg_path))
        seg_voxel_vol = np.prod(functions.read_dicom(seg_path).GetSpacing())*1e-3 #mL
        resampled_seg= read_resample(seg_path, abdose_path)

        for j in range(len(tumours_values)):
            tumour_key = tumours_values[j]
            act= quantification(spect_arr, seg_array, tumour_key) #24h quantification
            lesion_name=f"{patient}_{tumour_key}"
            if act > 0.01 and lesion_name != 'p17_25': #excludes lesions outside the FOV
                tumour_key = tumours_values[j]
                mean_dose= quantification_mean(abdose_arr, resampled_seg, tumour_key) * 7.4 #mean dose (Gy) with 7.4 GBq
                lesion_vol= lesion_volume(seg_array, tumour_key)*seg_voxel_vol
                if lesion_vol > 3: #excludes lesions with volume < 3 mL
                    lesion_name=f"{patient}_{tumour_key}"
                    if mean_dose > 100:
                        print(lesion_name)
                    patient_abdoses[lesion_name] = [mean_dose, lesion_vol]
                    if tumour_key == patient_lesion_id:
                        patient_selected_lesion[lesion_name] = [mean_dose, lesion_vol]

                else:
                    print(f"patient {patient} tumour key {tumour_key} less than 3 mL")
            else:
                print(f"patient {patient} tumour key {tumour_key} excluded - outside SPECT FOV")

        ads= [value for value in patient_abdoses.values()]
        print(f"{patient} mean AD across the evaluated lesions (n={len(ads)}): {np.mean(ads):.2f} Gy ({np.min(ads):.2f} - {np.max(ads):.2f})")


def plot_lesion_dose_volume(all_lesions:dict):

    doses = np.array([lesion[0] for lesion in all_lesions.values()], dtype=float)
    volumes = np.array([lesion[1] for lesion in all_lesions.values()], dtype=float)

    valid = (np.isfinite(doses) &np.isfinite(volumes) &(volumes >= 0))
    doses = doses[valid]
    volumes = volumes[valid]
    order = np.argsort(doses)[::-1]
    doses = doses[order]
    volumes = volumes[order]

    x = np.arange(len(doses))
    max_dose = doses.max() if len(doses) else 1
    max_volume = volumes.max() if len(volumes) else 1

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(x, doses, width=0.6, color="deepskyblue")
    ax1.set_ylim(-max_dose * 1.1, max_dose * 1.1)
    dose_ticks = np.linspace(0, max_dose, 4)
    ax1.set_yticks(dose_ticks)                     
    ax1.set_yticklabels([f"{v:.1f}" for v in dose_ticks])
    ax1.set_ylabel("Mean Absorbed Dose (Gy) ")
    ax1.yaxis.set_label_coords(-0.05, 0.75)     
    ax1.set_xticks([])
    ax1.set_xlim(-1, len(x))
    ax1.set_title("Mean Absorbed Dose (Gy) in Lesions Compared to Volumes")

    ax2 = ax1.twinx()
    ax2.bar(x, -volumes, width=0.6, color="coral")
    ax2.set_ylim(-max_volume * 1.1, max_volume * 1.1)
    vol_ticks = np.linspace(0, max_volume, 4)[1:]  
    ax2.set_yticks(-vol_ticks)                    
    ax2.set_yticklabels([f"{v:.0f}" for v in vol_ticks])

    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_ylabel("Volume (mL)")
    ax2.yaxis.set_label_coords(-0.05, 0.25)        
    ax1.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()

def abdose_lesions_volume(patients:list, tumours_keys:dict, true_dosimetry:bool):

    all_lesions_dose= {}
    selected_lesions_dose= {} #dict for the selected lesion in each patient
    for i in range(len(patients)):
        patient= patients[i]
        tumours_values= tumours_keys[patient]

        spect_24h_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_lu177.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"

        if true_dosimetry == False:
            abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21.nii"
        else:
            abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        
        patient_lesion_id= lesion_selection(seg_path, tumours_values, abdose_path) #id for the selected lesion in the optimisation algorithm
        
        spect_arr= sitk.GetArrayFromImage(functions.read_dicom(spect_24h_path))
        abdose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path)) #in Gy/GBq
        seg_array= sitk.GetArrayFromImage(functions.read_dicom(seg_path))
        seg_voxel_vol = np.prod(functions.read_dicom(seg_path).GetSpacing())*1e-3 #mL
        resampled_seg= read_resample(seg_path, abdose_path)

        for j in range(len(tumours_values)):
            tumour_key = tumours_values[j]
            act= quantification(spect_arr, seg_array, tumour_key) #24h quantification
            lesion_name=f"{patient}_{tumour_key}"
            if act > 0.01 and lesion_name != 'p17_25': #excludes lesions outside the FOV
                tumour_key = tumours_values[j]
                mean_dose= quantification_mean(abdose_arr, resampled_seg, tumour_key) * 7.4 #mean dose (Gy) with 7.4 GBq
                lesion_vol= lesion_volume(seg_array, tumour_key)*seg_voxel_vol
                if lesion_vol > 3: #excludes lesions with volume < 3 mL
                    lesion_name=f"{patient}_{tumour_key}"
                    if mean_dose > 100:
                        print(lesion_name)
                    all_lesions_dose[lesion_name] = [mean_dose, lesion_vol]
                    if tumour_key == patient_lesion_id:
                        selected_lesions_dose[lesion_name] = [mean_dose, lesion_vol]

                #else:
                 #   print(f"patient {patient} tumour key {tumour_key} less than 3 mL")
            #else:
             #   print(f"patient {patient} tumour key {tumour_key} excluded - outside SPECT FOV")

    ads= [values[0] for values in all_lesions_dose.values()]
    print(f"total lesions evaluated: {len(ads)}")
    print(f"mean AD across the evaluated lesions: {np.mean(ads):.2f} Gy - {np.mean(ads)/7.4:.2f} (Gy/GBq)")

    print(f"min AD across the evaluated lesions: {np.min(ads):.2f} Gy")
    print(f"max AD across the evaluated lesions: {np.max(ads):.2f} Gy")

    plot_lesion_dose_volume(all_lesions_dose)

def plot_lesion_diff_dose_volume(all_lesions:dict):

    doses = np.array([lesion[0] for lesion in all_lesions.values()], dtype=float)
    volumes = np.array([lesion[1] for lesion in all_lesions.values()], dtype=float)

    valid = (np.isfinite(doses) &np.isfinite(volumes) &(volumes >= 0))
    doses = doses[valid]
    volumes = volumes[valid]
    order = np.argsort(doses)[::-1]
    doses = doses[order]
    volumes = volumes[order]

    x = np.arange(len(doses))
    max_dose = doses.max() if len(doses) else 1
    max_volume = volumes.max() if len(volumes) else 1

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(x, doses, width=0.6, color="deepskyblue")
    ax1.set_ylim(-max_dose * 1.1, max_dose * 1.1)
    dose_ticks = np.linspace(0, max_dose, 4)
    ax1.set_yticks(dose_ticks)                     
    ax1.set_yticklabels([f"{v:.1f}" for v in dose_ticks])
    ax1.set_ylabel("Difference in Mean AD (%)")
    ax1.yaxis.set_label_coords(-0.05, 0.75)     
    ax1.set_xticks([])
    ax1.set_xlim(-1, len(x))
    ax1.set_title("Mean Absorbed Dose (Gy) in Lesions Compared to Volumes")

    ax2 = ax1.twinx()
    ax2.bar(x, -volumes, width=0.6, color="coral")
    #ax2.set_ylim(-max_volume * 1.1, max_volume * 1.1)
    vol_ticks = np.linspace(0, max_volume, 4)[1:]  
    ax2.set_yticks(-vol_ticks)                    
    ax2.set_yticklabels([f"{v:.0f}" for v in vol_ticks])

    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_ylabel("Volume (mL)")
    ax2.yaxis.set_label_coords(-0.05, 0.25)        
    ax1.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()

def plot_lesion_diff_dose_volume(all_lesions: dict):

    doses = np.array([lesion[0] for lesion in all_lesions.values()], dtype=float)
    volumes = np.array([lesion[1] for lesion in all_lesions.values()], dtype=float)

    valid = np.isfinite(doses) & np.isfinite(volumes) & (volumes > 0)
    doses = doses[valid]
    volumes = volumes[valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = np.where(doses >= 0, "deepskyblue", "coral")

    ax.scatter(
        volumes,
        doses,
        c=colors,
        s=60,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5
    )

    ax.set_xscale("log")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Volume (mL) - Log Scale")
    ax.set_ylabel("Difference in Mean AD (%)")

    ax.set_title("Difference in Mean AD vs Lesion Volume")

    ax.grid(alpha=0.25, which="both")

    plt.tight_layout()
    plt.show()

def abdose_lesions_diff(patients:list, tumours_keys:dict):

    all_lesions_dose= {}
    selected_lesions_dose= {} #dict for the selected lesion in each patient
    for i in range(len(patients)):
        patient= patients[i]
        tumours_values= tumours_keys[patient]

        spect_24h_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_lu177.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"

        abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21.nii"
        abdose_path_true= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        
        patient_lesion_id= lesion_selection(seg_path, tumours_values, abdose_path) #id for the selected lesion in the optimisation algorithm
        
        spect_arr= sitk.GetArrayFromImage(functions.read_dicom(spect_24h_path))
        abdose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path)) #in Gy/GBq
        abdose_arr_true= sitk.GetArrayFromImage(functions.read_dicom(abdose_path_true)) #in Gy/GBq

        seg_array= sitk.GetArrayFromImage(functions.read_dicom(seg_path))
        seg_voxel_vol = np.prod(functions.read_dicom(seg_path).GetSpacing())*1e-3 #mL
        resampled_seg= read_resample(seg_path, abdose_path)

        for j in range(len(tumours_values)):
            tumour_key = tumours_values[j]
            act= quantification(spect_arr, seg_array, tumour_key) #24h quantification
            lesion_name=f"{patient}_{tumour_key}"
            if act > 0.01 and lesion_name != 'p17_25': #excludes lesions outside the FOV
                tumour_key = tumours_values[j]
                mean_dose= quantification_mean(abdose_arr, resampled_seg, tumour_key) * 7.4 #mean dose (Gy) with 7.4 GBq
                true_mean_dose= quantification_mean(abdose_arr_true, resampled_seg, tumour_key) * 7.4
                lesion_vol= lesion_volume(seg_array, tumour_key)*seg_voxel_vol
                if lesion_vol > 3: #excludes lesions with volume < 3 mL
                    lesion_name=f"{patient}_{tumour_key}"

                    dif = 100*(mean_dose-true_mean_dose)/(true_mean_dose)

                    all_lesions_dose[lesion_name] = [dif, lesion_vol]
                    if tumour_key == patient_lesion_id:
                        selected_lesions_dose[lesion_name] = [dif, lesion_vol]

                #else:
                 #   print(f"patient {patient} tumour key {tumour_key} less than 3 mL")
            #else:
                #print(f"patient {patient} tumour key {tumour_key} excluded - outside SPECT FOV")

    ads= [values[0] for values in all_lesions_dose.values()]
    #print(f"total lesions evaluated: {len(ads)}")
    #print(f"mean AD across the evaluated lesions: {np.mean(ads):.2f} Gy - {np.mean(ads)/7.4:.2f} (Gy/GBq)")

    #print(f"min AD across the evaluated lesions: {np.min(ads):.2f} Gy")
    #print(f"max AD across the evaluated lesions: {np.max(ads):.2f} Gy")

    plot_lesion_diff_dose_volume(all_lesions_dose)