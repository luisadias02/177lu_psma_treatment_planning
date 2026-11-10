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
from platipy.imaging.dose.dvh import calculate_dvh, calculate_d_x, calculate_v_x


def quantification_mean(volume: np.array, seg: np.array, key:int):
    mask = seg == key
    mean = volume[mask].mean()
    return mean

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

def abdose_kidneys_init(patients:dict):

    for patient in list(patients.keys()):
        abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21.nii"
        seg_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"

        resampled_seg= read_resample(seg_path, abdose_path)
        abdose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_path))
        print(patient)

        for value in [2,3]:
            mean_dose_norm= quantification_mean(abdose_arr, resampled_seg, value) 
            mean_dose= mean_dose_norm* 7.4
            print(f"id {value}- mean AD: {mean_dose_norm :.2f} Gy/GBq - {mean_dose :.2f} Gy ")


def metrics(abdose_path:str, resampled_seg:np.array, values:list, best_act:float, threshold:float):
    
    abdose_img= functions.read_dicom(abdose_path)
    abdose_arr= (sitk.GetArrayFromImage(abdose_img))*best_act
    abdose_res_img= sitk.GetImageFromArray(abdose_arr)
    abdose_res_img.CopyInformation(abdose_img)

    dict_patient={}
    for value in values:
        mask = np.where(resampled_seg == value, 1, 0)
        if not mask.any():                      
            continue
        mask_img= sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(abdose_img)

        dose_bins, dvh_values = calculate_dvh(abdose_res_img, mask_img)
        
        mean= quantification_mean(abdose_arr, resampled_seg, value)
        d90_metric = np.interp(90 / 100, dvh_values[::-1], dose_bins[::-1])
        v = np.interp(threshold, dose_bins, dvh_values) * 100

        dict_patient[value] = [mean, d90_metric, v]

    return dict_patient


def kidney_metrics(patients:dict, oas:dict, threshold:float):

    all_values_mean_AD=[]
    all_values_d90= []
    all_values_v=[]
    for i in range(len(list(patients.keys()))):
        activities= oas[threshold]
        patient= list(patients.keys())[i]
        best_act= activities[i]
        abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        seg_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"
        
        resampled_seg= read_resample(seg_path, abdose_path)

        values=[2,3]
        if patient == "p17":
            values=[2]

        dict_patient= metrics(abdose_path, resampled_seg, values, best_act, threshold)
        right_kid_results=[value for value in dict_patient[2]]
        print(f"patient {patient}:")
        print(f"R. Kidney- Mean AD: {right_kid_results[0] :.2f} Gy, D90: {right_kid_results[1] :.2f} Gy, V{threshold}: {right_kid_results[2] :.2f}%")
        all_values_mean_AD.append(right_kid_results[0])
        all_values_d90.append(right_kid_results[1])
        all_values_v.append(right_kid_results[2])
        if 3 in values:
            left_kid_results=[value for value in dict_patient[3]]
            print(f"L. Kidney- Mean AD: {left_kid_results[0] :.2f} Gy, D90: {left_kid_results[1] :.2f} Gy, V{threshold}: {left_kid_results[2] :.2f}%")
            all_values_mean_AD.append(left_kid_results[0])
            all_values_d90.append(left_kid_results[1])
            all_values_v.append(left_kid_results[2])

    print(f"mean AD (Gy): {np.mean(all_values_mean_AD) :.2f} ({np.min(all_values_mean_AD):.2f} -- {np.max(all_values_mean_AD):.2f})")
    print(f"D90 (Gy): {np.mean(all_values_d90) :.2f} ({np.min(all_values_d90):.2f} -- {np.max(all_values_d90):.2f})")
    print(f"V{threshold} (%): {np.mean(all_values_v) :.2f} ({np.min(all_values_v):.2f} -- {np.max(all_values_v):.2f})")






    

