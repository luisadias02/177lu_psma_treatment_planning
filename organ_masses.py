import sys
import os
import csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from SimpleITK import ResampleImageFilter, Transform
import matplotlib.pyplot as plt
sys.path.append('C:\\ac225\\Ac-225')
#sys.path.append('/content/drive/MyDrive')
import functions


def import_files(ct_path, seg_path):

    reader= sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(ct_path)
    series_file_names = reader.GetGDCMSeriesFileNames(ct_path,series_ids[0])
    reader.SetFileNames(series_file_names)
    ct_image = reader.Execute()

    ct_arr= sitk.GetArrayFromImage(ct_image).astype(np.float32) 

    seg_img= functions.read_dicom(seg_path)
    seg_spacing= seg_img.GetSpacing() #mm
    seg_arr= sitk.GetArrayFromImage(seg_img)

    return ct_arr, seg_arr, seg_spacing


def hu_density_conversion(ct_array):

    threshold = 127.34355
    density = np.where(
            ct_array < threshold,
             (0.00102430 * ct_array + 1.02419109),  
                (0.000503556 * ct_array + 1.090504592))

    return density

def main(patients:list, csv_path:str):

    results = []

    for patient in patients:
        ct_path= f"F:\\champalimaud\\Ac225\\data_patients\\{patient}\\spects\\ct_spect"
        #ct_path= "F:\\champalimaud\\Ac225\\patients_phantoms\p11\\ct_spect.nrrd"
        seg_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\segmentations\\seg_spect_24h_w_tumours.nrrd"
        ct_arr, seg_arr, seg_spacing= import_files(ct_path, seg_path)

        density= hu_density_conversion(ct_arr)
        voxel_volume= seg_spacing[0]*seg_spacing[1]*seg_spacing[2]/ 1000 # conversion to cm3

        seg_keys={2:"right kidney", 3:"left kidney"}
        patient_result = {"patient": patient}

        for value, name in seg_keys.items():
            mask = seg_arr == value
            volume = mask.sum() * voxel_volume
            mean_density= density[mask].mean()
            mass= mean_density * volume / 1000 #in kilograms
            if name == "right kidney":
                patient_result["right_kidney_mass"] = mass
            elif name == "left kidney":
                patient_result["left_kidney_mass"] = mass

        results.append(patient_result)

    with open(csv_path, "w", newline="") as f:

        fieldnames = ["patient","right_kidney_mass", "left_kidney_mass" ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    

    











