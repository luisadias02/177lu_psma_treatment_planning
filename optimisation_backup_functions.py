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


def obj_fun(activity, vect_nlv_norm, vect_ptv_norm, thresh_nlv, thresh_ptv):
    ratio_nlv = np.sum(activity * vect_nlv_norm <= thresh_nlv) / vect_nlv_norm.size
    ratio_ptv = np.sum(activity * vect_ptv_norm >= thresh_ptv) / vect_ptv_norm.size
    return -(ratio_nlv + ratio_ptv)

def optimisation(oars_array, lesion_array):

    """
    This code reaches for the optimal activity required 
    """

    kidney_max = 3.5    # Gy, mean kidney dose ceiling (1 cycle)
    lesion_min = 1      # Gy, mean lesion dose floor   (1 cycle)

    A_min = lesion_min / lesion_array.mean()    # activity needed to reach the lesion floor
    A_max = kidney_max / oars_array.mean()       # activity at which the kidneys reach the ceiling

    #  - ratio_nlv stays 1 up to A_max and falls afterwards
    #  - ratio_ptv reaches 1 at A_min and stays there
    thresh_nlv = kidney_max * oars_array.max() / oars_array.mean()
    thresh_ptv = lesion_min * lesion_array.min() / lesion_array.mean()

    print(f"window: {A_min:.2f} to {A_max:.2f} GBq")

    # Infeasible: the lesion floor needs more activity than the kidneys allow -> kidney limit wins
    if A_min > A_max:
        best_act = A_max
        print("Note: lesion minimum not reachable within the kidney limit")
        print(f"optimised total kidney absorbed dose: {best_act * oars_array.mean() :.3f} Gy ")
        print(f"optimised total lesion absorbed dose: {best_act * lesion_array.mean() :.3f} Gy ")
        print(f"Optimal activity: {best_act:.3f} GBq  (defined by kidney limit)")
        print(" ")
        return best_act

    # GOLDEN SECTION SEARCH
    a = 0.0                          # start of interval (GBq)
    b = 20                           # end of interval (GBq)
    epsilon = 0.01                   # accuracy
    max_iter = 50                    # maximum number of iterations
    tau = (np.sqrt(5) - 1) / 2       # golden ratio coefficient, ~0.618
    k = 0

    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)

    f_x1 = obj_fun(x1, oars_array, lesion_array, thresh_nlv, thresh_ptv)
    f_x2 = obj_fun(x2, oars_array, lesion_array, thresh_nlv, thresh_ptv)

    fig, ax = plt.subplots()
    ax.plot(x1, -f_x1, 'b.')
    ax.plot(x2, -f_x2, 'b.')
    ax.set_xlim(a, b)

    while abs(b - a) > epsilon and k < max_iter:

        if f_x1 < f_x2:
            b = x2
            x2 = x1
            x1 = a + (1 - tau) * (b - a)
            f_x1 = obj_fun(x1, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            f_x2 = obj_fun(x2, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            ax.plot(x1, -f_x1, 'b.')
        else:
            a = x1
            x1 = x2
            x2 = a + tau * (b - a)
            f_x1 = obj_fun(x1, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            f_x2 = obj_fun(x2, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            ax.plot(x2, -f_x2, 'b.')
        k += 1

    if f_x1 < f_x2:
        best_act, best_val = x1, -f_x1
    else:
        best_act, best_val = x2, -f_x2

    print(f"optimised total kidney absorbed dose: {best_act * oars_array.mean() :.3f} Gy ")

    print(f"optimised total lesion absorbed dose: {best_act * lesion_array.mean() :.3f} Gy ")
    ax.axvspan(A_min, A_max, alpha=0.15, color='g')      # feasible window
    ax.plot(best_act, best_val, 'ro')
    ax.set_xlabel('Administered Activity (GBq)')
    ax.set_ylabel('Objective (NLV ratio + PTV ratio)')
    print(f'Optimal activity: {best_act:.3f} GBq  (objective = {best_val:.4f})')
    plt.show()

    return best_act

def obj_fun1(A, kid_vox, les_vox, thr_kid=3.5, t_les=None, lam=1.0):
    d_k = dose_at(A, kid_vox)
    d_l = dose_at(A, les_vox)
    shortfall = np.mean(np.clip(t_les - d_l, 0, None) ** 2) / t_les ** 2      # cold lesion voxels
    excess    = np.mean(np.clip(d_k - thr_kid, 0, None) ** 2) / thr_kid ** 2  # hot kidney voxels
    return shortfall + lam * excess                                            # minimise

def optimisation(oars_array, lesion_array):

def optimisation_2(oars_array, lesion_array, resampled_seg, lesion_id, abdose_map:str, a_ref=7.4, a_max=20, ax=None, title=None):

    dose_arr= sitk.GetArrayFromImage(functions.read_dicom(abdose_map))
    mask = resampled_seg == lesion_id
    mean_ad_lesion= np.mean(dose_arr[mask])*7.4 
    #mean_ad_lesion= np.median(dose_arr[mask])*7.4 

    thresh_nlv = 3.5                      # kidney maximum dose, Gy
    thresh_ptv = mean_ad_lesion           # lesion minimum dose, Gy

    kid_floor = min(dose_at(a_ref, oars_array).mean(), thresh_nlv)   # ceiling wins if a_ref exceeds it
    b = min(activity_for_mean_dose(thresh_nlv, oars_array), a_max)   # kidney mean <= 3.5 Gy
    a = min(activity_for_mean_dose(kid_floor, oars_array), b)        # kidney mean >= dose at a_ref
    print(f"Kidney window: {a:.2f} - {b:.2f} GBq")

    # GOLDEN SECTION SEARCH
    epsilon = 0.5
    max_iter = 200
    tau = (np.sqrt(5) - 1) / 2
    k = 0

    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)

    f_x1 = obj_fun(x1, oars_array, lesion_array, thresh_nlv, thresh_ptv)
    f_x2 = obj_fun(x2, oars_array, lesion_array, thresh_nlv, thresh_ptv)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots()

    ax.plot(x1, -f_x1, 'b.')
    ax.plot(x2, -f_x2, 'b.')
    ax.set_xlim(a - 0.5, b + 0.5)

    while abs(b - a) > epsilon and k < max_iter:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            x1 = a + (1 - tau) * (b - a)
            f_x1 = obj_fun(x1, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            f_x2 = obj_fun(x2, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            ax.plot(x1, -f_x1, 'b.')
        else:
            a = x1
            x1 = x2
            x2 = a + tau * (b - a)
            f_x1 = obj_fun(x1, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            f_x2 = obj_fun(x2, oars_array, lesion_array, thresh_nlv, thresh_ptv)
            ax.plot(x2, -f_x2, 'b.')
        k += 1

    if f_x1 < f_x2:
        best_act, best_val = x1, -f_x1
    else:
        best_act, best_val = x2, -f_x2

    #print(f"Optimal activity: {best_act:.3f} GBq  (objective = {best_val:.4f})")
    #print(f"Kidney mean dose: {dose_at(best_act, oars_array).mean():.2f} Gy")
    #print(f"Lesion mean dose: {dose_at(best_act, lesion_array).mean():.3f} Gy")

    ax.plot(best_act, best_val, 'ro', label=f"Optimum: {best_val:.3f}")
    ax.set_xlabel('Administered Activity (GBq)')
    ax.set_ylabel('Objective (NLV ratio + PTV ratio)')
    ax.legend(loc='best')
    if title:
        ax.set_title(title)
    if own_fig:          # only show when the function made its own figure
        plt.show()

    return best_act, best_val, mean_ad_lesion