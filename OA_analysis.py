import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import functions
import SimpleITK as sitk
from SimpleITK import ResampleImageFilter, Transform
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import bisect
from platipy.imaging.dose.dvh import calculate_dvh, calculate_d_x, calculate_v_x

def activity_effects(opt_activity:float, seg_path:str, abdose_path_true:str, abdose_path:str, tumour_keys:list, run:str):

    def load_data(dose_map_path:str, seg_path:str):

        dose_spect_true = np.clip(sitk.GetArrayFromImage(functions.read_dicom(dose_map_path)), 0, None) #in Gy/GBq

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

        resampled_seg = read_resample(seg_path, dose_map_path)
        return resampled_seg , dose_spect_true

    seg_arr, dose_arr_true= load_data(abdose_path_true, seg_path)
    dose_arr_sim= sitk.GetArrayFromImage(functions.read_dicom(abdose_path))

    patient_abdose={}
    if run == "kidneys":
        for value in [2,3]:
            mask = seg_arr == value
            mean_true = dose_arr_true[mask].mean()
            true_dose= (opt_activity) * mean_true
            mean_sim = dose_arr_sim[mask].mean()
            init_dose= (7.4) * mean_sim
            increase= ((true_dose/init_dose)-1)*100

            patient_abdose[value]= [init_dose, true_dose, increase]

    elif run == "tumours":
        for value in tumour_keys:
            mask = seg_arr == value
            mean_true = dose_arr_true[mask].mean()
            true_dose= (opt_activity) * mean_true
            mean_sim = dose_arr_sim[mask].mean()
            init_dose= (7.4) * mean_sim
            increase= ((true_dose/init_dose)-1)*100

            patient_abdose[value]= [init_dose, true_dose, increase]

    return patient_abdose, seg_arr

def cumulated_activity(opt_activity:float, abdose_path:str, seg_arr:np.array):

    """
    reads the ground truth absorbed dose map (in Gy/GBq) and calculates the cumulative 
    mean absorbed dose in the kidneys over 6 cycles with the determined optimal 
    activity (in GBq).
    """

    kidney_keys=[2,3]
    if "p17" in abdose_path:
        kidney_keys=[2]

    dose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path))
    dict_patient={}
    for value in kidney_keys:
        mask = seg_arr == value
        mean= dose_arr[mask].mean()
        std= dose_arr[mask].std()
        dose_6cycles= ((opt_activity) * mean)*6
        std_6cycles= ((opt_activity) * std)*6
        dict_patient[value] = [dose_6cycles, std_6cycles]

    return dict_patient

def cumulated_activity_1(opt_activity_1:float, opt_activity_2:float, abdose_path:str, seg_arr:np.array):

    act_cycle_1= opt_activity_1
    act_cycle_2= opt_activity_2

    kidney_keys=[2,3]
    if "p17" in abdose_path:
        kidney_keys=[2]

    dose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path))
    dict_patient={}
    for value in kidney_keys:
        mask = seg_arr == value
        mean= dose_arr[mask].mean()
        std= dose_arr[mask].std()
        dose_6cycles= ((act_cycle_1) * mean)*1 + ((act_cycle_2) * mean)*5
        std_6cycles= ((act_cycle_1) * std)*1 + ((act_cycle_2) * std)*5
        dict_patient[value] = [dose_6cycles, std_6cycles]

    return dict_patient

def main(opt_activity_1:float, opt_activity_2:float, seg_path:str, abdose_path_true, abdose_path, tumour_keys, run):

    patient_abdose_values, resampled_seg= activity_effects(opt_activity_1, seg_path, abdose_path_true, abdose_path, tumour_keys, run)

    if opt_activity_2 == None:
        dict_patient_cumulated= cumulated_activity(opt_activity_1, abdose_path_true, resampled_seg)
    else:
        dict_patient_cumulated= cumulated_activity_1(opt_activity_1, opt_activity_2, abdose_path_true, resampled_seg)

    return patient_abdose_values, dict_patient_cumulated, resampled_seg


### All patients analysis

def correlations(patients:list, opt_activities:list, abdose_paths:list, resampled_segs:list, lesion_ids:list, kid_values:list):

    dict_analysis={"A_increase":[], "mean_kidney":[], "median_kidney":[], "mean_lesion":[], "median_lesion":[],"mean_ratio":[], "median_ratio":[] }
    
    for i in range(len(patients)):

        opt_activity = opt_activities[i]
        abdose_path= abdose_paths[i]
        resampled_seg= resampled_segs[i]
        lesion_id= lesion_ids[i]
        kid_id= kid_values[i]
        activity_increase= ((opt_activity/7.4)-1) * 100
        dict_analysis["A_increase"].append(activity_increase)

        dose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path))*7.4

        #lesion metrics
        mask_lesion = resampled_seg == lesion_id
        arr= dose_arr[mask_lesion]
        mean_lesion= np.mean(arr)
        dict_analysis["mean_lesion"].append(mean_lesion)
        median_lesion= np.median(arr)
        dict_analysis["median_lesion"].append(median_lesion)

        #kidney_metrics
        mask_kidney= resampled_seg == kid_id
        arr= dose_arr[mask_kidney]
        mean_kid= np.mean(arr)
        dict_analysis["mean_kidney"].append(mean_kid)
        median_kid= np.median(arr)
        dict_analysis["median_kidney"].append(median_kid)

        mean_ratio= mean_kid/mean_lesion
        dict_analysis["mean_ratio"].append(mean_ratio)
        median_ratio= median_kid/median_lesion
        dict_analysis["median_ratio"].append(median_ratio)

    metrics= list(dict_analysis.keys())[1:]
    for metric in metrics:
        spear, p_value= spearmanr(dict_analysis["A_increase"], dict_analysis[metric])
        print(f"{metric}- correlation = {spear :.3f}, p_value= {p_value :.3f}")
        x = np.asarray(dict_analysis["A_increase"])
        y = np.asarray(dict_analysis[metric])
        valid = np.isfinite(x) & np.isfinite(y)
        x_valid = x[valid]
        y_valid = y[valid]
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(x_valid, y_valid)
        
    







