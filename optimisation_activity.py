import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import functions
import SimpleITK as sitk
from SimpleITK import ResampleImageFilter, Transform
import pandas as pd
from scipy.optimize import bisect
from platipy.imaging.dose.dvh import calculate_dvh, calculate_d_x, calculate_v_x
from kneed import KneeLocator


def tumour_metrics(volume: np.array, seg: np.array, key:int):
    mask = seg == key
    mean = volume[mask].mean()
    median =np.median(volume[mask])
    D80=np.percentile(volume[mask], 20)
    V35=np.mean(volume[mask] >= 3.5)

    return key, mean, median, D80, V35

def quantification_mean(volume: np.array, seg: np.array, key:int):
    mask = seg == key
    mean = volume[mask].mean()
    return mean

def quantification(volume: np.array, seg: np.array, key:int):
    mask = seg == key
    nvox = mask.sum()
    mean = volume[mask].mean()
    total= mean * nvox
    return total

def abdoses_at_activity(activity, dose_spect, seg, kidney_keys, tumours_keys):

    kidneys = {}
    for k in kidney_keys:
        mask = seg == k
        if mask.any():
            kidneys[k] = activity * dose_spect[mask].mean()

    lesions = {}
    for k in tumours_keys:
        mask = seg == k
        if mask.any():                      # lesions lost in the resampling are skipped
            lesions[k] = activity * dose_spect[mask].mean()

    return kidneys, lesions, sum(lesions.values())/len(lesions.values())

def calculate_dvh_lesion_kidneys(abdose_path:str, seg_resampled:np.array, lesion_id:int):

    abdose_img= functions.read_dicom(abdose_path)
    abdose_arr= (sitk.GetArrayFromImage(abdose_img))*7.4
    abdose_res_img= sitk.GetImageFromArray(abdose_arr)
    abdose_res_img.CopyInformation(abdose_img)

    values=[2, 3, lesion_id]
    if "p17" in abdose_path:
        values=[2, lesion_id]

    lesion_mean = quantification_mean(abdose_arr, seg_resampled, lesion_id)
    lesion_metrics= tumour_metrics(abdose_arr, seg_resampled, lesion_id)
    print(lesion_metrics)
    #lesion_median = quantification_median(abdose_arr, seg_resampled, lesion_id)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = {2: "Right kidney", 3: "Left kidney", lesion_id: "Lesion"}

    for value in values:
        mask = np.where(seg_resampled == value, 1, 0)
        if not mask.any():                      
            continue
        mask_img= sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(abdose_img)

        dose_bins, dvh_values = calculate_dvh(abdose_res_img, mask_img)
        ax.plot(dose_bins, dvh_values, linewidth=2, label=names[value])

    ax.axvline(lesion_mean, color="k", linestyle="--", linewidth=1.5,label=f"Lesion mean AD: {lesion_mean:.2f} Gy")    
    #ax.axvline(lesion_median, color="k", linestyle="--", linewidth=1.5,label=f"Lesion median AD: {lesion_median:.2f} Gy")    

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume")
    ax.set_title("Initial Dose Volume Histogram (DVH)")
    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def lesion_selection(spect_24h: str, seg_path:str, tumours_keys:list, abdose_map:str, resampled_seg:np.array, prints=True):

    #spect_arr= sitk.GetArrayFromImage(functions.read_dicom(spect_24h))
    abdose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_map))

    seg_img= functions.read_dicom(seg_path)
    seg_arr= sitk.GetArrayFromImage(seg_img)
    pixel_size= seg_img.GetSpacing()
    voxel_volume= (pixel_size[0]*pixel_size[1]*pixel_size[2])*(1e-3) #mL

    init_dose=1000
    lesion_id=0

    for key in tumours_keys:
        mask = seg_arr == key
        nvox = mask.sum()
        vol= nvox * voxel_volume
        if vol > 3:
            mean_dose_init= (quantification_mean(abdose_arr, resampled_seg, key))*7.4
            #mean_dose_init= (quantification_median(abdose_arr, resampled_seg, key))*7.4
            if mean_dose_init < init_dose and mean_dose_init > 0.1:
                init_uptake= mean_dose_init
                lesion_id= key

    if prints == True:
        print(f"selected lesion: {lesion_id} with a mean absorbed dose of {init_uptake :.3f} Gy")

    return lesion_id

def create_masks(resampled_seg:np.array, oar_key:list, lesions_keys:list):

    oar_map= np.zeros(resampled_seg.shape)
    lesion_map= np.zeros(resampled_seg.shape)

    for value in np.unique(resampled_seg):
        if value in oar_key:
            mask = np.where(resampled_seg == value, 1, 0)
            oar_map = oar_map + mask
        elif value in lesions_keys:
            mask = np.where(resampled_seg == value, 1, 0)
            lesion_map = lesion_map + mask

    return oar_map, lesion_map

def create_masks_all_lesions(resampled_seg:np.array, oar_key:list, lesions_keys:list):

    oar_map= np.zeros(resampled_seg.shape)
    lesion_map= np.zeros(resampled_seg.shape)

    for value in np.unique(resampled_seg):
        if value in oar_key:
            mask = np.where(resampled_seg == value, 1, 0)
            oar_map = oar_map + mask
        elif value in lesions_keys:
            mask = np.where(resampled_seg == value, 1, 0)
            lesion_map = lesion_map + mask

    return oar_map, lesion_map

def load_data(dose_map_path:str, spect_24h_path:str, seg_path:str, oar_key:list, a_admin:float, tumours_keys:list):
    # LOAD DOSE DISTRIBUTION, ROI MASKS AND ENTER ADMINISTERED ACTIVITY
    #dose_spect = sitk.GetArrayFromImage(functions.read_dicom(dose_map_path))
    dose_spect = np.clip(sitk.GetArrayFromImage(functions.read_dicom(dose_map_path)), 0, None)

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
    
    resampled_seg= read_resample(seg_path, dose_map_path)
    orig_quant= quantification_mean(dose_spect, resampled_seg, oar_key)
    #print(f"initial total kidney absorbed dose: {orig_quant* 7.4 :.3f} Gy")

    lesion_key= lesion_selection(spect_24h_path, seg_path, tumours_keys, dose_map_path, resampled_seg)
    #calculate_dvh_lesion_kidneys(dose_map_path, resampled_seg, lesion_key)
    #orig_quant_lesion= quantification_mean(dose_spect, resampled_seg, lesion_key)

    oars, tumours= create_masks(resampled_seg, oar_key, [lesion_key])

    # EXTRACT DOSE-ROI VECTORS 
    vect_nlv = dose_spect[oars != 0]
    vect_ptv = dose_spect[tumours != 0]

    #  NORMALIZE VECTORS (no need as the maps are in mGy/MBq)
    vect_nlv_norm = vect_nlv 
    vect_ptv_norm = vect_ptv 

    return vect_nlv_norm, vect_ptv_norm, dose_spect, resampled_seg, lesion_key

def dose_at(activity, vect):
    """Dose voxel map at a given activity."""
    return activity * vect

def activity_for_mean_dose(target, vect, a_hi=100.0):
    """Activity at which the mean dose of `vect` equals `target` Gy"""
    return bisect(lambda A: dose_at(A, vect).mean() - target, 0.0, a_hi)

def obj_fun_smooth(activity, vect_nlv_norm, vect_ptv_norm, thresh_nlv, thresh_ptv, lam):

    """
    this function aims to minimise the number of voxels within the lesion below the thresh_ptv value, which
    corresponds to the mean absorbed dose value (in Gy) at the 7.4 GBq injection, i.e. the mean absorbed dose in 
    the lesion increases necessarily. 

    it also minimises the number of voxels above the 3.5 Gy threshold (thresh_nlv) in order to stay within the
    23 Gy limit over 6 cycles.

    A weight (lam) was attributed to the kidney term and the shortfall and excess correspond to the weighted mean 
    square errors. the np.clip sets to zero any voxel on the right side of its limit (i.e. lesion voxels above 
    the target and kidney voxels below the 3.5 Gy threshold), hence, only violating voxels contribute 
    to the optimisation.

    The aim of the optimisation is to minimise the term (shortfall + lam * excess), with better results being 
    at optimums close to zero. 

    """

    d_kid = dose_at(activity, vect_nlv_norm)
    d_les = dose_at(activity, vect_ptv_norm)

    shortfall = np.mean(np.clip(thresh_ptv - d_les, 0, None)**2) / thresh_ptv **2    # cold lesion voxels below the threshold 
    excess    = np.mean(np.clip(d_kid - thresh_nlv, 0, None)**2) / thresh_nlv **2    # kidney voxels above the threshold
    return shortfall + lam * excess  # to minimise                                               

def optimisation_3(oars_array:np.array, lesion_array:np.array, resampled_seg:np.array, lesion_id:int, abdose_map:str,
                   a_ref=7.4, a_max=20, ax=None, title=None, lam=0.5, plot=True, kidney_thresh=None):

    dose_arr = sitk.GetArrayFromImage(functions.read_dicom(abdose_map))
    mask = resampled_seg == lesion_id
    #mean_ad_lesion = np.mean(dose_arr[mask]) * 7.4
    mean_ad_lesion = np.mean(dose_arr[mask]) * 7.4

    thresh_nlv = kidney_thresh                     # kidney maximum dose, Gy
    thresh_ptv = mean_ad_lesion           # target dose per lesion voxel, Gy

    kid_floor = min(dose_at(a_ref, oars_array).mean(), thresh_nlv)
    b = min(activity_for_mean_dose(thresh_nlv, oars_array), a_max)
    a = min(activity_for_mean_dose(kid_floor, oars_array), b)
    if plot == True:
        print(f"Kidney window: {a:.2f} - {b:.2f} GBq")

    f = lambda A: obj_fun_smooth(A, oars_array, lesion_array, thresh_nlv, thresh_ptv, lam)

    # GOLDEN SECTION SEARCH (minimisation)
    epsilon = 0.05
    max_iter = 200
    tau = (np.sqrt(5) - 1) / 2
    k = 0

    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f_x1 = f(x1)
    f_x2 = f(x2)

    own_fig = plot and ax is None          
    if own_fig:
        fig, ax = plt.subplots()

    if plot== True:
        acts = np.linspace(a, b, 100)
        ax.plot(acts, [f(A) for A in acts], 'k-', alpha=0.3)
        ax.plot(x1, f_x1, 'b.')
        ax.plot(x2, f_x2, 'b.')
        ax.set_xlim(a - 0.5, b + 0.5)

    while abs(b - a) > epsilon and k < max_iter:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            x1 = a + (1 - tau) * (b - a)
            f_x1 = f(x1)
            f_x2 = f(x2)
            if plot== True:
                ax.plot(x1, f_x1, 'b.')
        else:
            a = x1
            x1 = x2
            x2 = a + tau * (b - a)
            f_x1 = f(x1)
            f_x2 = f(x2)
            if plot== True:
                ax.plot(x2, f_x2, 'b.')
        k += 1

    if f_x1 < f_x2:
        best_act, best_val = x1, f_x1
    else:
        best_act, best_val = x2, f_x2

    if plot == True:
        ax.plot(best_act, best_val, 'ro', label=f"Optimum: {best_val:.3f}")
        ax.set_xlabel('Administered Activity (GBq)')
        ax.set_ylabel('Objective (lesion shortfall + lam x kidney excess)')
        ax.legend(loc='best')
        if title:
            ax.set_title(title)
        if own_fig:
            plt.show()

    return best_act, best_val, mean_ad_lesion

def kidney_selection_optimisation(dict_results:dict):

    """selects the optimal activity based on the total lower kidney mean absorbed dose"""

    if list(dict_results.keys()) == [2,3]:
        print("evaluating the 2 kidneys...")
        results_rk= dict_results[2]
        otm_rk =  results_rk[1]
        kidneys_mean_rk= results_rk[3]+results_rk[4]
        results_lk= dict_results[3] 
        otm_lk =  results_lk[1]
        kidneys_mean_lk= results_lk[3]+results_lk[4]
        if kidneys_mean_rk > kidneys_mean_lk:
        #if otm_rk < otm_lk:
            best_act= results_lk[0]
            best_otm= otm_lk
            kid_value= 3
        else:
            best_act= results_rk[0]
            best_otm= otm_rk
            kid_value= 2

    else:
        print("evaluating for 1 kidney only...")
        for value in [2,3]:
            if value in dict_results.keys():
                result= dict_results[value]
                best_act= result[0]
                best_otm= result[1]
                kid_value= value

    return best_act, best_otm, kid_value

def lesion_increase(abdose_map:str, abdose_map_true:str, seg:np.array, tumour_keys:list, a_opt:float):

    abdose_arr= np.clip(sitk.GetArrayFromImage(functions.read_dicom(abdose_map)), 0, None) #Gy/GBq
    initial_arr = abdose_arr * 7.4

    if abdose_map_true == None:
        opt_arr= abdose_arr * a_opt
    else:
        true_abdose_arr= np.clip(sitk.GetArrayFromImage(functions.read_dicom(abdose_map_true)), 0, None)
        opt_arr= true_abdose_arr * a_opt

    initial_mean= []
    opt_mean= []
    increases=[]

    for value in tumour_keys:
        mean_init = quantification_mean(initial_arr, seg, value)
        mean_opt = quantification_mean(opt_arr, seg, value)
        if np.isfinite(mean_init) and np.isfinite(mean_opt) and mean_init > 0:
            initial_mean.append(mean_init)
            opt_mean.append(mean_opt)
            if (mean_opt / mean_init - 1) * 100 < 1000:
                increases.append((mean_opt / mean_init - 1) * 100) #excludes lesions outside the SPECT FOV
        else:
            print(f"skipped lesion {value}: voxels in seg = {(seg == value).sum()}, mean_init = {mean_init}")

    if not initial_mean:
        return np.nan, np.nan, np.nan

    return np.mean(initial_mean), np.mean(opt_mean), np.mean(increases)

def main(abdose_path:str, seg_path:str, spect_24h_path:str, a_admin, patient_keys, lam:float, plot=True, prints=True, 
         kidney_thresh=None):

    kidney_keys = [2] if "p17" in abdose_path else [2, 3]
    kidney_names = {2: "Right", 3: "Left"}
    kidneys_abdose = {}

    loaded = {}
    for value in kidney_keys:
        loaded[value] = load_data(abdose_path, spect_24h_path, seg_path, [value], a_admin, patient_keys)

    if plot == True:
        calculate_dvh_lesion_kidneys(abdose_path, loaded[2][3], loaded[2][4])
    
    if plot ==True:
        fig, axes = plt.subplots(1, len(kidney_keys), figsize=(6 * len(kidney_keys), 4),
                                 sharey=True, squeeze=False)
        axes = axes.ravel()
    else:
        axes = [None] * len(kidney_keys)

    for ax, value in zip(axes, kidney_keys):
        vect_nlv, vect_ptv, dose_spect, resampled_seg, lesion_id = loaded[value]

        best_act, best_val, mean_ad_lesion = optimisation_3(
            vect_nlv, vect_ptv, resampled_seg, lesion_id, abdose_path,
            ax=ax, title=f"{kidney_names[value]} kidney", lam = lam, plot=plot, kidney_thresh= kidney_thresh)

        kidneys, lesions, lesion_total = abdoses_at_activity(
            best_act, dose_spect, resampled_seg, kidney_keys, patient_keys)

        other = [k for k in kidney_keys if k != value]

        kidneys_abdose[value] = [best_act,best_val,kidneys[value] + (kidneys[other[0]] if other else 0),kidneys[value],
            (kidneys[other[0]] if other else None),mean_ad_lesion,lesion_total,[d for k, d in lesions.items()],]

    if plot == True:
        fig.tight_layout()
        plt.show()   

    mean_ad_lesion = kidneys_abdose[kidney_keys[0]][5]
    best_act_2, best_otm_2, kid_value = kidney_selection_optimisation(kidneys_abdose)
    print(f"optimal activity: {best_act_2:.2f} GBq")
    #print(f"optimum: {best_otm_2:.3f}")

    kidneys, lesions, lesion_total = abdoses_at_activity(best_act_2, dose_spect, resampled_seg, kidney_keys, patient_keys)
    if prints == True:
        for k in kidney_keys:
            if k in kidneys:
                print(f"{kidney_names[k]} kidney: {kidneys[k]:.2f} Gy")
        print(f"lesion_mean: {lesion_total:.2f} Gy")

    return best_act_2, mean_ad_lesion, best_otm_2, loaded[2][4], kid_value, resampled_seg

def main_evaluation_lambda(abdose_path: str,seg_path: str,spect_24h_path: str,a_admin: float,patient_keys: list,patient: str,
                           kidney_thresh:float):

    lambda_values = np.round(np.arange(0.1, 1, 0.05), 2)

    dose_image = functions.read_dicom(abdose_path)
    seg_image = functions.read_dicom(seg_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(dose_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampled_seg = resampler.Execute(seg_image)
    seg_arr = sitk.GetArrayFromImage(resampled_seg)

    dose_map = sitk.GetArrayFromImage(dose_image)
    kidney_keys = [2] if "p17" in abdose_path else [2, 3]
    results = []

    for lam in lambda_values:
        print(f"{patient}: evaluating lambda = {lam:.2f}")
        best_act, ad_lesion, best_obj, lesion_id, kid_value, resampled_seg = main(abdose_path,seg_path,spect_24h_path,a_admin,
            patient_keys,lam=lam,plot=False,prints=False, kidney_thresh= kidney_thresh)

        dose_arr = dose_map * best_act
        row = {"patient": patient,"lambda": float(lam),"best_activity_GBq": best_act,"best_objective": best_obj,"lesion_id": lesion_id,}

        for kidney in kidney_keys:
            kidney_mask = dose_arr[seg_arr == kidney]
            row[f"kidney_{kidney}_mean_Gy"] = np.mean(kidney_mask)
            row[f"kidney_{kidney}_max_Gy"] = np.max(kidney_mask)
            row[f"kidney_{kidney}_above_3.5"] = (np.mean(kidney_mask > kidney_thresh) * 100)
            row[f"kidney_{kidney}_p90_Gy"] = np.percentile(kidney_mask, 90)

        mask_les = seg_arr == lesion_id
        lesion_mask = dose_arr[mask_les]
        thresh_lesion = np.mean(dose_arr[mask_les])/best_act * 7.4

        row["lesion_reference_mean_Gy"] = thresh_lesion
        row["lesion_mean_Gy"] =  np.mean(lesion_mask)
        row["cold_below_100"] = (np.mean(lesion_mask < thresh_lesion) * 100)
        row["cold_below_90"]  = (np.mean(lesion_mask < 0.90 * thresh_lesion) * 100)
        row["cold_below_80"]  = (np.mean(lesion_mask < 0.80 * thresh_lesion) * 100)
        results.append(row)

    return pd.DataFrame(results)

def choose_lambda(results:pd.DataFrame):

    df = results.copy()

    kidney_cols = [c for c in ["kidney_2_max_Gy", "kidney_3_max_Gy"] if c in df.columns]
    df["kidney_worst_max_Gy"] = df[kidney_cols].max(axis=1, skipna=True)

    kidney_p90_cols = [c for c in ["kidney_2_p90_Gy", "kidney_3_p90_Gy"]if c in df.columns]
    df["kidney_worst_p90_Gy"] = df[kidney_p90_cols].max(axis=1, skipna=True)


    summary = (df.groupby("lambda").agg(n_patients=("patient", "nunique"),
            activity_mean=("best_activity_GBq", "mean"),
            activity_sd=("best_activity_GBq", "std"),
            cold90_mean=("cold_below_90", "mean"),
            cold90_sd=("cold_below_90", "std"),
            kidney_max_mean=("kidney_worst_max_Gy", "mean"),
            kidney_max_sd=("kidney_worst_max_Gy", "std"),
            kidney_max_p90=("kidney_worst_max_Gy",lambda x: x.quantile(0.90)),
            kidney_p90_mean=("kidney_worst_p90_Gy", "mean"), #patients individual quartile
            kidney_p90_sd=("kidney_worst_p90_Gy", "std"),
            kidney_2_above_mean=("kidney_2_above_3.5", "mean"),
            kidney_2_above_sd=("kidney_2_above_3.5", "std"),
            kidney_3_above_mean=("kidney_3_above_3.5", "mean"),
            kidney_3_above_sd=("kidney_3_above_3.5", "std"),).reset_index().sort_values("lambda"))

    metrics = [("best_activity_GBq", "activity_mean", "activity_sd","Optimal administered activity (GBq)"),
    ("cold_below_90", "cold90_mean", "cold90_sd","Lesion voxels below 90% of target (%)"),
    ("kidney_worst_max_Gy", "kidney_max_mean", "kidney_max_sd","Worst kidney maximum dose (Gy)")]

    kidney_metrics = [(2, "kidney_2_above_3.5", "Kidney 2"),(3, "kidney_3_above_3.5", "Kidney 3")]

    summary = summary.sort_values("lambda").copy()
    summary["coverage90_mean"] = 100 - summary["cold90_mean"]

    x_raw = summary["cold90_mean"]
    x_raw= summary["coverage90_mean"] 
    #y_raw = summary["kidney_max_mean"]
    y_raw = summary["kidney_p90_mean"]

    x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min()) #data normalisation
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) #data normalisation

    s = summary.sort_values("coverage90_mean")
    kl = KneeLocator(s["coverage90_mean"].values, s["kidney_p90_mean"].values, S= 0.3,
                    curve="convex", direction="increasing")
    #interp_method="polynomial"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, marker="o")

    for (_, row), xi, yi in zip(summary.iterrows(), x, y):
        ax.annotate(f'{row["lambda"]:.2f}',(xi, yi),xytext=(-25, 4),textcoords="offset points",fontsize=9)

    if kl.knee is not None:
        i_knee = (summary["coverage90_mean"] - kl.knee).abs().idxmin()     
        lam_knee = summary.loc[i_knee, "lambda"]
        ax.plot(x.loc[i_knee], y.loc[i_knee], "o", color="red", markersize=10, zorder=4,
                label=f"Knee: λ = {lam_knee:.2f}")
        ax.legend(loc="upper left", )
        print(f"knee: lambda = {lam_knee:.2f} | coverage = {kl.knee:.1f}% | kidney max = {kl.knee_y:.2f} Gy")
    else:
        print("kneed found no knee: try a smaller S (e.g. 0.5) or interp_method='polynomial'")

    ax.set_xlabel("Normalised mean lesion voxels with at least 90% of target (%)")
    ax.set_ylabel("Normalised mean of patient-level worst kidney maximum (Gy)")
    ax.set_title("Cohort values across lambda values")
    ax.grid(alpha=0.3)
    plt.show()



    



