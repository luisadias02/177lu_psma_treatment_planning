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
from platipy.imaging.dose.dvh import calculate_dvh


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
        #v = np.interp(threshold, dose_bins, dvh_values) * 100

        dict_patient[value] = [mean, d90_metric]

    mask_bool = np.isin(resampled_seg, [2,3]) 
    mask_img = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
    mask_img.CopyInformation(abdose_img) 
    dose_bins, dvh_values = calculate_dvh(abdose_res_img, mask_img)
    v = np.interp(threshold, dose_bins, dvh_values) * 100

    dict_patient["kid_v23"] = [v]

    return dict_patient

def kidney_metrics(patients:dict, oas:dict, threshold:float):

    """
    Returns the true kidney mean absorbed dose (Gy), D90 (Gy) and V_threshold (%) from the 
    dose-volume histogram of the kidney for a single cycle. Code runs for both kidneys separately. 

    """

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
        print(f"R. Kidney- Mean AD: {right_kid_results[0] :.2f} Gy, D90: {right_kid_results[1] :.2f} Gy")
        all_values_mean_AD.append(right_kid_results[0])
        all_values_d90.append(right_kid_results[1])
        #all_values_v.append(right_kid_results[2])
        if 3 in values:
            left_kid_results=[value for value in dict_patient[3]]
            print(f"L. Kidney- Mean AD: {left_kid_results[0] :.2f} Gy, D90: {left_kid_results[1] :.2f} Gy")
            all_values_mean_AD.append(left_kid_results[0])
            all_values_d90.append(left_kid_results[1])
            #all_values_v.append(left_kid_results[2])

        v_23 = [v for v in dict_patient["kid_v23"]]
        print(f"V{threshold}: {v_23[0] :.2f}%")
        all_values_v.append(v_23)


    print(f"mean AD (Gy): {np.mean(all_values_mean_AD) :.1f} ({np.min(all_values_mean_AD):.1f} -- {np.max(all_values_mean_AD):.1f})")
    print(f"D90 (Gy): {np.mean(all_values_d90) :.1f} ({np.min(all_values_d90):.1f} -- {np.max(all_values_d90):.1f})")
    print(f"V{threshold} (%): {np.mean(all_values_v) :.1f} ({np.min(all_values_v):.1f} -- {np.max(all_values_v):.1f})")


    patient_names = list(patients.keys())
    results = {thr: {"oa": [], "mean_ad": [], "d90": [], "v": [], "pid": [], "pid_k": []}
               for thr in dict_thresholds_oas}

    for i, patient in enumerate(patient_names):
        abdose_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"
        resampled_seg = read_resample(seg_path, abdose_path)

        values = [2] if patient == "p17" else [2, 3]

        for threshold, activities in dict_thresholds_oas.items():
            best_act = activities[i]
            dict_patient = metrics(abdose_path, resampled_seg, values, best_act, threshold)

            results[threshold]["oa"].append(best_act)              # one OA per patient
            results[threshold]["pid"].append(i)                    # patient index (oa, v)

            for k in values:                                       # one value per kidney
                mean_ad, d90 = list(dict_patient[k])[:2]
                results[threshold]["mean_ad"].append(mean_ad)
                results[threshold]["d90"].append(d90)
                results[threshold]["pid_k"].append(i)              # patient index (mean_ad, d90)

            v = dict_patient["kid_v23"][0]
            results[threshold]["v"].append(v)

    panels = [("oa",      "Optimal activity (GBq)"),
              ("mean_ad", "Kidney mean AD (Gy)"),
              ("d90",     r"Kidney $D_{90}$ (Gy)"),
              ("v",       r"Kidney $V_{T}$ (%)")]

    pid_key = {"oa": "pid", "v": "pid", "mean_ad": "pid_k", "d90": "pid_k"}

    labels = [str(k) for k in results.keys()]
    positions = np.arange(1, len(labels) + 1)
    pcolors = plt.cm.tab20(np.arange(len(patient_names)) % 20)     # one color per patient
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (key, ylabel) in zip(axes, panels):
        data = [np.asarray(results[t][key], dtype=float) for t in results]

        bp = ax.boxplot(data, positions=positions, widths=0.55,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.3), capprops=dict(linewidth=1.3))

        for patch in bp["boxes"]:                                  # neutral boxes
            patch.set_facecolor("0.9")
            patch.set_edgecolor("black")

        for pos, t in zip(positions, results):
            vals = np.asarray(results[t][key], dtype=float)
            pids = np.asarray(results[t][pid_key[key]])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(pos + jitter, vals, s=35, c=pcolors[pids],
                       edgecolor="black", linewidth=0.4, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{l}\n(n={len(d)})" for l, d in zip(labels, data)])
        ax.set_xlabel("Mean kidney AD threshold (Gy)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # one shared legend with one color per patient
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=pcolors[j],
                          markeredgecolor="black", label=name)
               for j, name in enumerate(patient_names)]
    fig.legend(handles=handles, title="Patient", loc="center right")

    fig.tight_layout(rect=(0, 0, 0.9, 1))                          # room on the right for the legend
    plt.show()

def kidney_metrics_threshold(patients:dict, dict_thresholds_oas:dict):

    patient_names = list(patients.keys())
    results = {thr: {"oa": [], "mean_ad": [], "d90": [], "v": [], "pid": [], "pid_k": []}
               for thr in dict_thresholds_oas}

    anon = {"p11": "P1",  "p15": "P2",  "p24": "P3",  "p32": "P4","p33": "P5",  "p17": "P6",  "p23": "P7",  "p25": "P8","p28": "P9",  "p31": "P10", "p13": "P11", "p12": "P12"}

    for i, patient in enumerate(patient_names):
        abdose_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        seg_path = f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"
        resampled_seg = read_resample(seg_path, abdose_path)

        values = [2] if patient == "p17" else [2, 3]

        for threshold, activities in dict_thresholds_oas.items():
            best_act = activities[i]
            dict_patient = metrics(abdose_path, resampled_seg, values, best_act, threshold)
            results[threshold]["oa"].append(best_act)              # one OA per patient
            results[threshold]["pid"].append(i)                    # patient index (oa, v)
            for k in values:                                       # one value per kidney
                mean_ad, d90 = list(dict_patient[k])[:2]
                results[threshold]["mean_ad"].append(mean_ad)
                results[threshold]["d90"].append(d90)
                results[threshold]["pid_k"].append(i)              # patient index (mean_ad, d90)

            v = dict_patient["kid_v23"][0]
            results[threshold]["v"].append(v)

    panels = [("oa", "Optimal activity (GBq)"), ("mean_ad", "Kidney mean AD (Gy)"),
              ("d90", r"Kidney $D_{90}$ (Gy)"), ("v", r"Kidney $V_{Thresh}$ (%)")]

    pid_key = {"oa": "pid", "v": "pid", "mean_ad": "pid_k", "d90": "pid_k"}

    thr_list = list(results.keys())
    labels = [str(k) for k in thr_list]
    positions = np.arange(1, len(labels) + 1)
    pcolors = plt.cm.tab20(np.arange(len(patient_names)) % 20)
    rng = np.random.default_rng(0)

    with plt.rc_context({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "legend.title_fontsize": 8.5, "axes.linewidth": 0.7}):

        fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5 * 0.75))
        axes = axes.ravel()

        for ax, (key, ylabel) in zip(axes, panels):
            data = [np.asarray(results[t][key], dtype=float) for t in thr_list]
            mat = np.vstack(data)
            pids = np.asarray(results[thr_list[0]][pid_key[key]])

            bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5), whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9), boxprops=dict(linewidth=0.9))

            for patch in bp["boxes"]:
                patch.set_facecolor("0.9")
                patch.set_edgecolor("black")

            jitter = rng.uniform(-0.12, 0.12, size=mat.shape[1])

            # light line per patient (per kidney in the mean AD and D90 panels) across thresholds
            for j in range(mat.shape[1]):
                ax.plot(positions + jitter[j], mat[:, j], color=pcolors[pids[j]],
                        alpha=0.4, linewidth=0.7, zorder=2)

            for t_idx, pos in enumerate(positions):
                ax.scatter(pos + jitter, mat[t_idx], s=14, c=pcolors[pids],
                           edgecolor="black", linewidth=0.3, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{l}\n(n={len(d)})" for l, d in zip(labels, data)])
            ax.set_xlabel("Mean kidney AD threshold (Gy)", fontsize='small')
            ax.set_ylabel(ylabel, fontsize='small')
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        order = sorted(range(len(patient_names)), key=lambda j: int(anon[patient_names[j]][1:]))
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=5,
                              markerfacecolor=pcolors[j], markeredgecolor="black",
                              markeredgewidth=0.3, label=anon[patient_names[j]])
                   for j in order]
        fig.legend(handles=handles, title="Patient", loc="lower center", ncol=6,
                   frameon=False, columnspacing=1.2)

        fig.tight_layout(rect=(0, 0.1, 1, 1))
        plt.show()


def metrics_6cycles(abdose_path:str, resampled_seg:np.array, values:list, best_act_1:float, best_act_others:float, threshold:float):
    
    abdose_img= functions.read_dicom(abdose_path)
    abdose_arr= (sitk.GetArrayFromImage(abdose_img))*best_act_1 + (sitk.GetArrayFromImage(abdose_img))*5 * best_act_others
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
        #v = np.interp(threshold, dose_bins, dvh_values) * 100
    
        dict_patient[value] = [mean, d90_metric]

    mask_bool = np.isin(resampled_seg, [2,3]) 
    mask_img = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
    mask_img.CopyInformation(abdose_img) 
    dose_bins, dvh_values = calculate_dvh(abdose_res_img, mask_img)
    v = np.interp(threshold, dose_bins, dvh_values) * 100

    dict_patient["bilateral"] = [v]

    return dict_patient

def plot_v23(dict_cumulated_patients, threshold=23, limit=30):
    patients = list(dict_cumulated_patients.keys())
    v = [dict_cumulated_patients[p]["bilateral"][0] for p in patients]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(patients, v, color="tab:blue", edgecolor="black")
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.axhline(limit, color="red", linestyle="--", label=f"{limit}% limit")

    ax.set_ylabel(f"V{threshold} Gy (% of total kidney volume)")
    ax.set_title(f"Kidney V{threshold} Gy over 6 cycles")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def kidney_6cycles(patients:dict, oas:dict, thresholds:float):

    """
    Returns the true kidney mean absorbed dose (Gy), D90 (Gy) and V_threshold (%) from the 
    dose-volume histogram of the kidney. Code runs for both kidneys separately. 

    """

    all_values_mean_AD=[]
    all_values_d90= []
    all_values_v=[]

    dict_cumulated_patients = {} 

    for i in range(len(list(patients.keys()))):

        threshold_cycle_1= thresholds[0]
        threshold_others= thresholds[1]

        activities_1= oas[threshold_cycle_1]
        activities_others= oas[threshold_others]

        total_threshold= threshold_cycle_1 + 5*threshold_others

        patient= list(patients.keys())[i]
        best_act_1= activities_1[i]
        best_act_others= activities_others[i]

        abdose_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes_1\\lu177\\{patient}_abdose_2p21_true.nii"
        seg_path= f"F:\\champalimaud\\Ac225\\patients_phantoms\\{patient}\\recon_volumes\\seg_spect_24h_w_tumours_resampled.nrrd"
        
        resampled_seg= read_resample(seg_path, abdose_path)
        values=[2,3]
        if patient == "p17":
            values=[2]

        dict_patient= metrics_6cycles(abdose_path, resampled_seg, values, best_act_1, best_act_others, 23)
        right_kid_results=[value for value in dict_patient[2]]
        #print(f"patient {patient}:")
        #print(f"R. Kidney- Mean AD: {right_kid_results[0] :.2f} Gy, D90: {right_kid_results[1] :.2f} Gy, V{total_threshold}: {right_kid_results[2] :.2f}%")
        all_values_mean_AD.append(right_kid_results[0])
        all_values_d90.append(right_kid_results[1])

        if 3 in values:
            left_kid_results=[value for value in dict_patient[3]]
            #print(f"L. Kidney- Mean AD: {left_kid_results[0] :.2f} Gy, D90: {left_kid_results[1] :.2f} Gy, V{total_threshold}: {left_kid_results[2] :.2f}%")
            all_values_mean_AD.append(left_kid_results[0])
            all_values_d90.append(left_kid_results[1])

        all_values_v.append(dict_patient["bilateral"])
        dict_cumulated_patients[patient] = dict_patient

    print(f"mean AD (Gy): {np.mean(all_values_mean_AD) :.1f} ({np.min(all_values_mean_AD):.1f} -- {np.max(all_values_mean_AD):.1f})")
    print(f"D90 (Gy): {np.mean(all_values_d90) :.1f} ({np.min(all_values_d90):.1f} -- {np.max(all_values_d90):.1f})")
    print(f"V23 (%): {np.mean(all_values_v):.1f} ({np.min(all_values_v):.1f} -- {np.max(all_values_v):.1f})")

    plot_v23(dict_cumulated_patients)

    #return dict_cumulated_patients



    
        

        







    

