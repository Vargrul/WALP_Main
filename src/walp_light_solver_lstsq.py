import numpy as np
from numpy.ma import maximum
from scipy import linalg
from scipy import optimize 
from random import gauss
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp

import walp_misc_functions
import walp_validation
import walp_re_render

def a(rho, sm, N=None, L=None, dot=None):
    if(len(rho.shape) == 1):
        rho = np.reshape(rho, (rho.shape[0], 1))

    if dot is None and (N is not None and L is not None):
        # Prepare a vector of the Light Vector for the dot product calculations
        L_ar = np.ones(N.shape) * L
        # Calculate the dot product for each normal in regards to the light vector
        dot = walp_misc_functions.mp_dot(N, L_ar, maximum=0)
        # Reshape the result to make it match the other parameters for vector wise calculations
        dot = np.reshape(dot, (dot.shape[0],1))
    elif dot is None:
        raise ValueError("Input N and L is needed when no dot array is given!")

    return rho * dot * sm
    # return rho * np.cos(np.maximum(0,dot)) * ~sm

def b(rho, c):
    if(len(rho.shape) == 1):
        rho = np.reshape(rho, (rho.shape[0], 1))

    return rho*c

def est_lstsq(rho, p, Sm, c, N=None, L=None, dot=None, mask=None):
    # Check for inputs
    if dot is None and (N is None or L is None):
        raise ValueError("Please input EITHER dot OR N and L")

    # remove samples without data
    if mask is not None:
        p = p * mask[:, 0]
        rho = rho * mask[:, 0]

    # calculate a and b for the equation x1*Is + x2*Ia = p
    x1 = a(rho, Sm, N=N, L=L, dot=dot)
    x2 = b(rho, c)

    # Concatenate the columns to the A side og the linear equation
    A = np.concatenate((x1, x2), axis=1)

    # Find the X(the unknown variables) in the linear equation
    x  = linalg.lstsq(A,p)
    # x = optimize.nnls(A,p)

    return x

def est_lstsq_rgb(rho, p, Sm, c, N=None, L=None, dot=None, mask=None):
    # Check for inputs
    if dot is None and (N is None or L is None):
        raise ValueError("Please input EITHER dot OR N and L")

    # precalculate the dot product for use over all three color channels
    L_ar = np.ones(N.shape) * L

    if dot is None:
        # Calculate the dot product for each normal in regards to the light vector
        dot = walp_misc_functions.mp_dot(N, L_ar, maximum=0)
    
    # Reshape the result to make it match the other parameters for vector wise calculations
    dot = np.reshape(dot, (dot.shape[0],1))

    # Estimate separate color channels
    xr = est_lstsq(rho[:,0], p[:,0], Sm, c, L=L_ar, dot=dot, mask=mask)
    xg = est_lstsq(rho[:,1], p[:,1], Sm, c, L=L_ar, dot=dot, mask=mask)
    xb = est_lstsq(rho[:,2], p[:,2], Sm, c, L=L_ar, dot=dot, mask=mask)

    # Concatenate to rgb light sources
    l1 = [xr[0][0], xg[0][0], xb[0][0]]
    l2 = [xr[0][1], xg[0][1], xb[0][1]]

    # The returned residual is empty (and no 0)  sometimes. This is to ensure a value
    res = [[ xr[1] if type(xr[1]) == np.float64 else 0],
        [ xg[1] if type(xg[1]) == np.float64 else 0],
        [ xb[1] if type(xb[1]) == np.float64 else 0]]

    return l1, l2, res

if __name__ == "__main__":
    
    # data_path = 'D:/OneDrive - Aalborg Universitet/Projects/World as a Light Probe/data/markup_street/daylight_cycle_long_2/'
    data_path = 'D:/OneDriveAau/OneDrive - Aalborg Universitet/Projects/World as a Light Probe/data/markup_street/daylight_cycle_long_2/'

    rho, p, shadow_mask, N, rgb, c = walp_misc_functions.load_data(data_path, 'VRaySamplerInfo/top_scene.VRaySamplerInfo.0000.jpg', 
        'VRayDiffuseFilter/top_scene.VRayDiffuseFilter.0000.jpg', 
        'VRayShadows/top_scene.VRayShadows.0842.jpg', 
        "top_scene0842.jpg", 
        "cloud_visibility/top_scene.jpg")
    L = walp_misc_functions.load_light_vector(data_path + "sunvector/sun_vector.txt")[842]

    sun, sky = est_lstsq_rgb(rho, p, shadow_mask, c, N=N, L=L)

    print("Sun value:\t{}\t{}\t{}\t".format(*sun))
    print("Sky value:\t{}\t{}\t{}\t".format(*sky))

    walp_validation.validate_data(rgb, sun, sky, rho, shadow_mask, c, N=N, L=L)

    # Plot some stuff for visualization purposes
    # plot = plt.subplot(3,1,1).imshow(diffuse_reflectance_img)
    # plot = plt.subplot(3,1,2).imshow(rgb_img)
    # plot = plt.subplot(3,1,3).imshow(np.reshape(shadow_mask, (diffuse_reflectance_img.shape[0], diffuse_reflectance_img.shape[1])), cmap='gray')
    # plt.show()
