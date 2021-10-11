import numpy as np
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import pickle
# from skimage.transform import resize
import skimage.transform, skimage.color

from numpy.linalg import norm
from scipy import ndimage

import walp_openexr_loader as exr_loader
import walp_misc_functions as m_func



def save_pckl_data(name, var):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(var, f)

def load_pckl_data(name):
    if name[-4:] == ".pkl":
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

def mp_dot(v1, v2, negative_values=False, maximum=None):
    # Calculate dot product for all points using multiprocessing
    # with mp.Pool(processes = mp.cpu_count()) as p:
    #     dot = p.starmap(np.dot, zip(v1, v2))
    if not (v2[0]>0).any()  and not negative_values:
        dot = np.zeros(v2.shape[0])
    else:
        dot = np.einsum('ij, ij->i', v1, v2)
    
    if maximum is not None:
        dot = np.maximum(maximum, dot)
    return np.array(dot)

def precalc_dot(files_normals, file_lightvector, dir_savelocation, is_openexr = False, start=0, stop=-1, step=1):
    list_L = load_vector_from_file(file_lightvector)[start:stop:step]

    proc = 0
    max_proc = len(list_L)
    for f_n, L in zip(files_normals[start:stop:step], list_L):
        proc = proc + 1
        print("Calculating dot: {:6} of {:6}".format(proc, max_proc))
        if is_openexr:
            N = conver_to_normals(exr_loader.get_img_channel(f_n, ['VRayNormalWorld'])[0], direct=False, normalise=True)
            pass
        else:
            N = load_normals_from_uint8("", f_n)
        L_ar = np.ones(N.shape) * L

        dot = mp_dot(N,L_ar, maximum=0)
        dot = np.reshape(dot, (dot.shape[0],1))
        save_pckl_data(dir_savelocation + "\\dot_{:05}".format(proc+start), dot)

    # return dots

def load_normals_from_uint8(data_folder, img_normal_name):
    # Get the normal image with normal in WORLD coordinates!
    normals_img = np.array(Image.open(data_folder + img_normal_name))
    # Convert to float
    normals_img = normals_img/255.
    N = conver_to_normals(normals_img)
    
    return N

def conver_to_normals(in_img, direct=False, normalise=False):
    out_img = in_img
    if not direct:
        out_img = (in_img - 0.5) * 2
    # Reshape image to a single colum
    out_img = np.reshape(out_img, (out_img.shape[0] * out_img.shape[1], 3))
    if normalise:
        norm_scales = np.linalg.norm(out_img, axis=1)
        norm_scales = np.column_stack((norm_scales,norm_scales,norm_scales))
        out_img = out_img / norm_scales
        out_img = np.nan_to_num(out_img)
    return out_img

def load_data_exr(exr_filename, sky_vis_filename, dot_filename = None):
    # Load all channels
    channels = exr_loader.get_img_channel(exr_filename, ["VRayNormalWorld", "VRayDiffuseFilter", "VRayRawShadow", "", "A", "VRayWorldPoint"], [True, True, True, True, False, True])
    normals_img = channels[0]
    diffuse_reflectance_img = channels[1]
    shadow_img = np.mean(channels[2], axis=2)
    rgb_img = channels[3]
    alpha_img = channels[4]
    alpha_bool = alpha_img > 0.5
    alpha_bool = ndimage.binary_erosion(alpha_bool, iterations=5)
    world_pos = conver_to_normals(channels[5], direct=True)
    sky_vis_img = exr_loader.get_all_channels(sky_vis_filename)

    # Convert to grayscale
    sky_vis_img = skimage.color.rgb2gray(sky_vis_img)

    # If p is larger then simulated data, resize p
    if rgb_img.shape is not sky_vis_img.shape:
        rgb_img = skimage.transform.resize (rgb_img, sky_vis_img.shape[0:2])

    rho = np.reshape(diffuse_reflectance_img, (diffuse_reflectance_img.shape[0]*diffuse_reflectance_img.shape[1], 3))
    shadow_mask = 1-np.reshape(shadow_img, (shadow_img.shape[0]*shadow_img.shape[1], 1))
    sky_vis = np.reshape(sky_vis_img, (sky_vis_img.shape[0]*sky_vis_img.shape[1], 1))
    p = np.reshape(rgb_img, (rgb_img.shape[0]*rgb_img.shape[1], 3))
    alpha_mask = np.reshape(alpha_bool, (alpha_img.shape[0]*alpha_img.shape[1], 1))
    N = m_func.conver_to_normals(normals_img, direct=False, normalise=True)

    if dot_filename is not None:
        dot = load_pckl_data(dot_filename)
    else:
        dot = None

    return rho, p, shadow_mask, N, rgb_img, sky_vis, dot, alpha_mask, world_pos

def load_data_p(file_path, size=None):
    # Load real data
    rgb_img = np.array(Image.open(file_path))/255.0

    if size is not None:
        rgb_img = skimage.transform.resize(rgb_img, size)

    p = np.reshape(rgb_img, (rgb_img.shape[0]*rgb_img.shape[1], 3))
    return p

def load_data(data_folder, img_normal_filename, img_diffuse_filename, img_shadow_filename, img_apperance_filename, img_sky_vis_filename, dot_filename = None, alpha_filename=None):
    # Relative path for the dataset used + the prefix of the file

    N = load_normals(data_folder, img_normal_filename)

    # load the diffuse reflectance image
    diffuse_reflectance_img = np.array(Image.open(data_folder + img_diffuse_filename))/255.0
    # Load the 
    rho = np.reshape(diffuse_reflectance_img, (diffuse_reflectance_img.shape[0]*diffuse_reflectance_img.shape[1], 3))

    # Load and 
    shadow_img = 1-np.array(Image.open(data_folder + img_shadow_filename).convert('L'))/255.0
    # shadow_mask_temp = shadow_img > 10
    # print(np.sum(shadow_mask_temp))
    shadow_mask = np.reshape(shadow_img, (shadow_img.shape[0]*shadow_img.shape[1], 1))

    sky_vis_img = np.array(Image.open(data_folder + img_sky_vis_filename).convert('L'))/255.0
    sky_vis = np.reshape(sky_vis_img, (sky_vis_img.shape[0]*sky_vis_img.shape[1], 1))

    # Create/load light vector (the Y dimension needs to be flipped to match 3ds max!)
    # l_vec = np.array([0.739,0.672,0.049])
    # Normalize light vector - if input is not normalized
    # L = np.linalg.norm(l_vec) * l_vec

    # For this test *ONLY* this is simulated to have a known Is and Ia. This should otherwise be the pixel value of the image
    # rgb_img = walp_re_render.sim_data(rho, [200.0,200.0,200.0], [0, 0, 0], shadow_mask, (*shadow_img.shape, 3), N=N, L=L,)
    
    # Load real data
    rgb_img = np.array(Image.open(data_folder + img_apperance_filename))/255.0
    p = np.reshape(rgb_img, (rgb_img.shape[0]*rgb_img.shape[1], 3))

    if dot_filename is not None:
        dot = load_pckl_data(dot_filename)
    else:
        dot = None

    if alpha_filename is not None:
        alpha_img = np.array(Image.open(data_folder + alpha_filename).convert('L'))/255.0
        alpha_mask = np.reshape(alpha_img, (alpha_img.shape[0]*alpha_img.shape[1], 1))
    else:
        alpha_mask = None

    return rho, p, shadow_mask, N, rgb_img, sky_vis, dot, alpha_mask

def load_vector_from_file(file_name, normalise=True):
    L = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for l in lines:
            l = l[1:-2].split(',')
            l = np.array(list(map(float, l)))
            if normalise:
                l = l / np.linalg.norm(l)
            L.append(l)
    return L



def load_data_sequence(data_folder, img_normal_name_list, img_diffuse_name_list, img_shadow_name_list, img_apperance_name_list, file_lightvector, img_sky_vis_list):
    rho = []
    p = []
    shadow_mask = []
    N = []
    rgb_img = []
    sky_vis = []
    for img_normal_name, img_diffuse_name, img_shadow_name, img_apperance_name, img_sky_vis in zip(img_normal_name_list, img_diffuse_name_list, img_shadow_name_list, img_apperance_name_list, img_sky_vis_list):
        tmp_rho, tmp_p, tmp_shadow_mask, tmp_N, tmp_rgb_img, tmp_sky_vis = load_data(data_folder, img_normal_name, img_diffuse_name, img_shadow_name, img_apperance_name, img_sky_vis)

        rho.append(tmp_rho)
        p.append(tmp_p)
        shadow_mask.append(tmp_shadow_mask)
        N.append(tmp_N)
        rgb_img.append(tmp_rgb_img)
        sky_vis.append(tmp_sky_vis)

    L = load_vector_from_file(file_lightvector)

    return rho, p, shadow_mask, N, L, rgb_img, sky_vis