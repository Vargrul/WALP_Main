import numpy as np
from PIL import Image
from numpy.lib.index_tricks import AxisConcatenator
from numpy.linalg import norm
from numpy.ma import maximum

import walp_misc_functions as m_func
import walp_openexr_loader as oexr_loader

# Calculate areas where specularity might be problematic and remove them.
# Return changed alpha mask
# Simple = Phong Specularity
# Complex = ?TBD
def specularity_danger(normals, world_pos, cam_pos, light_pos, threshold=0.1, pre_mask=None):

    L = compute_l(light_pos, shape = world_pos.shape)
    R = compute_r(normals, L)
    V = compute_v(cam_pos, world_pos)
    S = calc_phong_specularity(R, V, 256)
    S_alpha = threshold_specularity(S)

    ret_mask = pre_mask & np.invert(S_alpha)

    return ret_mask

def threshold_specularity(S, threshold=0.1):
    ret_bool = np.reshape(S > threshold, (S.shape[0],1))

    return ret_bool

def calc_phong_specularity(R, V, a):
    dot = m_func.mp_dot(R, V, negative_values=True, maximum = 0)
    S = dot ** a

    return S

# compute the reflection vector "r" for each point
def compute_r(N, L):
    dot = m_func.mp_dot(N, L, negative_values=True)
    # dot = np.tensordot(N, L, axes=([1],[1]))
    dot = np.reshape(dot, (dot.shape[0],1))
    R = 2*dot*N-L
    return R

def compute_v(CV, WP):
    # Calculate view vectors
    V = CV - WP

    # Normalize view vectors
    norm_scales = np.linalg.norm(V, axis=1)
    norm_scales = np.column_stack((norm_scales,norm_scales,norm_scales))
    V = V / norm_scales
    V = np.nan_to_num(V)

    return V

def compute_l(LP, WP = None, shape = None) -> np.array:
    """Generate a normalized per-pixel light vector map and calculate the lightvector if
    the lightsource is not directionalself.

    Args:
        LP (Light Position): The Position of the lightsource.
        WP (World Position, optional): The world position for each pixel in the image. Defaults to None.
        shape (Shape of the image, optional): The shape of the image, this is needed to generate a per pixel. Defaults to None.

    Returns:
        np.array: Returns an array with normalized per-pixel Light Vectors.
    """

    # Create a per-point light vector map
    if shape is not None:
        L = np.zeros(shape=shape)
        L = L+LP
    else:
        L = LP

    # Calculate direction vectors towards the lightsource if the lightsource
    # is not directional
    if WP is not None:
        L = LP - WP

    # Normalize view vectors
    norm_scales = np.linalg.norm(L, axis=1)
    norm_scales = np.column_stack((norm_scales,norm_scales,norm_scales))
    L = L / norm_scales
    L = np.nan_to_num(L)
    return L

# Using rasac to remove bad data
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
def ransac():
    pass

# Visualize the data
def overlay_points_on_img():
    pass

def vis_specularity(S, img_shape):
    c = np.minimum(255, np.reshape(S*255.0, img_shape[0:2]))
    p = np.dstack((c,c,c)).astype(np.uint8)

    return Image.fromarray(p)

def vis_R(R, img_shape):
    if  R[:,0].max() > 255 or  R[:,0].min() < 0:
        r_unscale = R[:,0] - R[:,0].min()
    else:
        r_unscale = R[:,0]

    if  R[:,1].max() > 255 or  R[:,1].min() < 0:
        g_unscale = R[:,1] - R[:,1].min()
    else:
        g_unscale = R[:,1]

    if  R[:,2].max() > 255 or  R[:,2].min() < 0:
        b_unscale = R[:,2] - R[:,2].min()
    else:
        b_unscale = R[:,2]

    # This is to ensure the data does not overflow a uint8 - This can happen due to over exposure in the "ground truth" image
    scale = np.amax([r_unscale.max(), g_unscale.max(), b_unscale.max()])
    r = np.minimum(255, np.reshape(r_unscale*(255.0/scale), img_shape[0:2]))
    g = np.minimum(255, np.reshape(g_unscale*(255.0/scale), img_shape[0:2]))
    b = np.minimum(255, np.reshape(b_unscale*(255.0/scale), img_shape[0:2]))

    p = np.dstack((r,g,b)).astype(np.uint8)

    return Image.fromarray(p)

if __name__ == "__main__":
    
    data_path = 'E:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\markup_street\\DS5'
    cam_vector_filename = '\\cam_vec.txt'
    light_vector_filename = '\\sun_vector.txt'
    img_app_img_filename = '\\top_scene1079.jpg'
    img_exr_filename = '\\vray_raw\\top_scene.1079.exr'
    # img_normals_filename = '\\VRaySamplerInfo\\top_scene.VRaySamplerInfo.1079.jpg'

    
    # load images
    app_img = np.array(Image.open(data_path + img_app_img_filename))
    normals = m_func.conver_to_normals(oexr_loader.get_img_channel(data_path + img_exr_filename, 'VRayNormalWorld'), direct=False, normalise=True)
    world_pos = m_func.conver_to_normals(oexr_loader.get_img_channel(data_path + img_exr_filename, 'VRayWorldPoint'), direct=True)
    cam_pos = m_func.load_vector_from_file(data_path + cam_vector_filename, normalise=False)
    light_pos = m_func.load_vector_from_file(data_path + light_vector_filename, normalise=False)

    L = compute_l(light_pos[1079], shape = world_pos.shape)
    R = compute_r(normals, L)
    V = compute_v(cam_pos[1079], world_pos)
    S = calc_phong_specularity(R, V, 256)
    S_alpha = threshold_specularity(S)

    n_img = vis_R(normals, app_img.shape)
    s_img = vis_specularity(S, app_img.shape)
    r_img = vis_R(R, app_img.shape)
    l_img = vis_R(L, app_img.shape)
    v_img = vis_R(V, app_img.shape)
    S_alpha_img = vis_specularity(S_alpha, app_img.shape)

    Image.fromarray(app_img).show()
    n_img.show()
    s_img.show()
    S_alpha_img.show()
    r_img.show()
    l_img.show()
    v_img.show()

    print()