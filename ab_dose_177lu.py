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

### DOSIMETRY OF 177LU-PSMA
### Map computation, dose map and dose results

def ct_conversion(ct_path:str, tia_map:str):

    ct_image= functions.read_dicom(ct_path)
    tia_map= functions.read_dicom(tia_map) 

    def read_resample(image, ref_image):
        resampler = ResampleImageFilter()
        resampler.SetReferenceImage(ref_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(Transform())
        resampled_img= resampler.Execute(image)
        return resampled_img
    
    resampled_ct_img= read_resample(ct_image, tia_map)
    resampled_ct_arr= sitk.GetArrayFromImage(resampled_ct_img)

    resampled_ct_arr = np.clip(resampled_ct_arr, -1000, None)

    threshold = 127.34355
    density = np.where(resampled_ct_arr < threshold,
            (0.00102430 * resampled_ct_arr + 1.02419109),  
            (0.000503556 * resampled_ct_arr + 1.090504592))

    density = np.clip(density, 0.2, None)   # ~lung density

    return density

def map_computation(dict_tia: dict, dict_tia_tumours:dict, volume_path: str, seg_path:str, output_path: str):

    labels= dict_tia.keys()
    spect_img= functions.read_dicom(volume_path)
    pixel_size= spect_img.GetSpacing()
    spect_arr= sitk.GetArrayFromImage(spect_img)
    size= spect_arr.shape
    seg_arr = sitk.GetArrayFromImage(functions.read_dicom(seg_path))

    total_labels= [0, 1, 2, 3, 5] + list(dict_tia_tumours.keys())

    spect_arr_norm= spect_arr/((pixel_size[0]*pixel_size[1]*pixel_size[2])*(1e-3))
    total_array = np.zeros((size[0], size[1], size[2]))
    new_seg = np.zeros_like(seg_arr)
    for label in total_labels:
        new_seg[seg_arr == label] = label

    for i in total_labels:
        mask = new_seg == i
        if i in [0, 1,2,3,5]:
            tia_rescaled= dict_tia[i]/((pixel_size[0]*pixel_size[1]*pixel_size[2])*(1e-3)) #voxel TIA conservation for later resampling 
        else: 
            tia_rescaled= dict_tia_tumours[i]/((pixel_size[0]*pixel_size[1]*pixel_size[2])*(1e-3))

        voi_values = spect_arr_norm[mask]
        voi_sum = np.sum(voi_values)
        conv_voi= (voi_values / voi_sum) * tia_rescaled 
        total_array[mask] += conv_voi

    total_image= sitk.GetImageFromArray(total_array)
    total_image.SetDirection(functions.read_dicom(volume_path).GetDirection())
    total_image.SetOrigin(functions.read_dicom(volume_path).GetOrigin())
    total_image.SetSpacing(functions.read_dicom(volume_path).GetSpacing())

    sitk.WriteImage(total_image, output_path)
    #print('TIA map saved in:', output_path)

    return output_path

def resampling_2p21(tia_map_path:str, output_path: str):

    original_tia_image= functions.read_dicom(tia_map_path)
    original_spacing= original_tia_image.GetSpacing()
    #print('original spacing:', original_spacing)
    original_size= original_tia_image.GetSize()
    #print('original size:', original_size)

    new_spacing= (2.21, 2.21, 2.21)
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(original_tia_image.GetOrigin())
    resampler.SetOutputDirection(original_tia_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform()) 

    resampled_tia_map= resampler.Execute(original_tia_image)
    sitk.WriteImage(resampled_tia_map, output_path)
    
    #print('Resampled TIA map saved in:', output_path)

def kernel_computation():
    # This kernel corresponds to the 2.21 mm voxel size, for 177Lu in soft tissue.
    # The kernel S-Values are in mGy/(MBq*s)

    data = np.loadtxt("s_values_lu177_2p21.txt", delimiter='\t', skiprows=2) # this kernel corresponds to the 2.21 mm voxel size, for 177Lu in soft tissue. The kernel S-Values are in mGy/(MBq*s)

    s_values = data[:, 3]
    dim = int(round(len(s_values) ** (1/3)))
    s_values_img = s_values.reshape((dim, dim, dim))

    s_values_all = np.zeros((2*dim - 1, 2*dim - 1, 2*dim - 1))

    for i in range(2*dim - 1):
        ii = abs(i - (dim - 1))
        for j in range(2*dim - 1):
            jj = abs(j - (dim - 1))
            for k in range(2*dim - 1):
                kk = abs(k - (dim - 1))
                s_values_all[i, j, k] = s_values_img[ii, jj, kk]

    return s_values_all

def density_correction(ct_path:str, tia_map_path:str, ab_dose_map):
    density_kernel=  1.04 #g/cm3
    ct_density_arr= ct_conversion(ct_path, tia_map_path)
    #ct_density_arr= sitk.GetArrayFromImage(ct_density_img)
    abdose_corrected= ab_dose_map*(density_kernel/ct_density_arr)

    return abdose_corrected

def abdose(activity_map_path:str, tia_values:dict, tia_values_tumours:dict, seg_path:str, kernel, ct_path:str, output_path:str):

    tia_map_image= functions.read_dicom(activity_map_path)
    image_spacing= tia_map_image.GetSpacing()
    total_labels= list(tia_values.keys())+ list(tia_values_tumours.keys())

    def read_resample(seg_path:str, recon_path:str):
        def read_dicom(path):
                image= functions.read_dicom(path)
                #print(image.GetSize())
                #print(image.GetSpacing())
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
    
    resampled_seg= read_resample(seg_path, activity_map_path)
    np_tia_map = sitk.GetArrayFromImage(tia_map_image)
    np_tia_map= np_tia_map*(image_spacing[0]*image_spacing[1]*image_spacing[2]*(1e-3)) #scaled to voxel volume

    for value in total_labels:
        if value in list(tia_values.keys()):
            expected_tia= tia_values[value]
        else:
            expected_tia= tia_values_tumours[value]

        mask= resampled_seg == value
        conv= np_tia_map[mask]
        current_tia= np.sum(conv)
        if current_tia > 0 and np.isfinite(expected_tia) and current_tia != expected_tia :
            scale_factor = expected_tia / current_tia
            np_tia_map[mask] *= scale_factor

    corrected_image = sitk.GetImageFromArray(np_tia_map)
    corrected_image.CopyInformation(tia_map_image)
    sitk.WriteImage(corrected_image, activity_map_path)

    ab_dose_map = signal.fftconvolve(np_tia_map, kernel, mode='same')
    corrected_abdose_map = density_correction(ct_path, activity_map_path, ab_dose_map)
    ab_dose_map_corrected = corrected_abdose_map / 7400

    ab_dose_image= sitk.GetImageFromArray(ab_dose_map_corrected)
    ab_dose_image.SetSpacing(tia_map_image.GetSpacing())
    ab_dose_image.SetOrigin(tia_map_image.GetOrigin())
    ab_dose_image.SetDirection(tia_map_image.GetDirection())
    sitk.WriteImage(ab_dose_image, output_path)

    keys= {2:'right kidney', 3:'left kidney'} #segmentation keys if TotalSegmentator is used
    
    dose_results= {}
    for value in keys.keys():                
        mask = resampled_seg == value
        conv_value = ab_dose_map_corrected[mask]

        total_dose= np.sum(conv_value) #in mGy/MBq
        mean_dose = np.mean(conv_value) 
        std_dose = np.std(conv_value)

        dose_results[keys[value]]= [total_dose, mean_dose,std_dose]

        for value in tia_values_tumours.keys():            
            mask = resampled_seg == value
            lesion_volume= mask.sum()*np.prod(ab_dose_image.GetSpacing())/1000 
            conv_value = ab_dose_map_corrected[mask]
            if conv_value.size == 0:                      
                dose_results[f"tumour {value}"] = [np.nan, np.nan, np.nan, np.nan]
                continue

            total_dose = np.sum(conv_value)              
            mean_dose = np.mean(conv_value)
            std_dose = np.std(conv_value)

            dose_results[f"tumour {value}"] = [total_dose, mean_dose, std_dose, lesion_volume]

    return dose_results


def main(patient:str, tia_values:list, tumours_keys: list, true_dosimetry= False):

    if true_dosimetry==True:
        print("performing dosimetry with the ground truth values")
    else:
        print("performing dosimetry with the measured values")

    spect_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_lu177.nii"
    seg_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"
    ct_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\{patient}_ct.nrrd"

    dict_tia_oars= {1:  tia_values[3], 2:  tia_values[0], 3: tia_values[1], 5: tia_values[2], 0: tia_values[4]}
    tia_values_tumours= tia_values[5]
    dict_tia_tumours={}
    for key, tia in zip(tumours_keys, tia_values_tumours):
        dict_tia_tumours[key] = tia

        
    output_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_tia_map.nii"
    output_path_res= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_tia_map_res_2p21.nii"

    if true_dosimetry==True:
        output_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_tia_map_true.nii"
        output_path_res= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_tia_map_res_2p21_true.nii"


    output_tia_path= map_computation(dict_tia_oars, dict_tia_tumours, spect_path, seg_path, output_path)
    resampling_2p21(output_tia_path,  output_path_res)

    kernel= kernel_computation()
    output_path_abdose_map= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21.nii"

    if true_dosimetry==True:
        output_path_abdose_map= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        
    dose_results_patient= abdose(output_path_res, dict_tia_oars, dict_tia_tumours, seg_path, kernel, ct_path, output_path_abdose_map)

    return dose_results_patient

    












