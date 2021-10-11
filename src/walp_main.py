import os
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.type_check import real

import walp_light_solver_lstsq
import walp_validation
import walp_misc_functions
import walp_errors
import walp_data_reduction as walp_dr

def create_input_lists(path_main, path_normals, path_diffuse, path_shadow, path_rgb, path_cloud_vis, path_dots = None, path_alpha = None):
    # Create list of paths for a loop
    if path_dots is None:
        paths = (path_main + path_normals, path_main + path_diffuse, path_main + path_shadow, path_main + path_rgb, path_main + path_cloud_vis)
    else:
        paths = (path_main + path_normals, path_main + path_diffuse, path_main + path_shadow, path_main + path_rgb, path_main + path_cloud_vis, path_main + path_dots)

    if path_alpha is not None:
        paths = paths + (path_main + path_alpha,)

    files = []
    for p in paths:
        with os.scandir(p) as entries:
            files.append([entry.path for entry in entries if entry.is_file()])

    # This could be optimised, the same image is loaded for each frame
    for i in range(len(files)):
        if len(files[i]) == 1:
            files[i] = [files[i][0] for j in range(len(files[3]))]

    if path_dots is not None and path_alpha is not None:
        return files[0], files[1], files[2], files[3], files[4], files[5], files[6]
    elif path_dots is not None or path_alpha is not None:
        return files[0], files[1], files[2], files[3], files[4], files[5]
    elif path_dots is None and path_alpha is None:
        return files[0], files[1], files[2], files[3], files[4]


def create_input_list_exr(path_main, path_exr, path_cloud_vis, path_dots, path_p=None):
    if path_p is not None:
        paths = (path_main + path_exr, path_main + path_cloud_vis, path_main + path_dots, path_main + path_p)
    else:
        paths = (path_main + path_exr, path_main + path_cloud_vis, path_main + path_dots)
    

    files = []
    for p in paths:
        with os.scandir(p) as entries:
            files.append([entry.path for entry in entries if entry.is_file()])

    # This could be optimised, the same image is loaded for each frame
    for i in range(len(files)):
        if len(files[i]) == 1:
            files[i] = [files[i][0] for j in range(len(files[0]))]

    if path_p is not None:
        return files[0], files[1], files[2], files[3]
    else:
        return files[0], files[1], files[2]

# def run_sequence(files_exr=None, files_normals=None, files_diffuse=None, files_shadows=None, files_rgb=None, file_sunvector=None, files_sky_vis=None, files_alpha=None, files_dots=None, start=0, stop=-1, step=1, dots=None, use_alpha=False):

#     if files_alpha is not None:
#         use_alpha = True
#     if use_alpha is True and files_exr is None and files_alpha is None:
#         raise walp_errors.FuncInputError("There where no alpha files given!")

#     rgb_entered = files_normals is not None or files_diffuse is not None or files_shadows is not None or files_rgb is not None or file_sunvector is not None or files_sky_vis is not None
#     if files_exr is None and not rgb_entered:
#         raise walp_errors.FuncInputError("Neither RGB file path(s) or exr file path(s) was given.")
#     elif files_exr is not None and rgb_entered:
#         raise walp_errors.FuncInputError("Both RGB file path(s) or exr file path(s) was given. Please only input one or the other.")

#     if rgb_entered:
#         return run_sequence_rgb(files_normals, files_diffuse, files_shadows, files_rgb, file_sunvector, files_sky_vis, files_dots=None, start=0, stop=-1, step=1, dots=None, files_alpha=None)
#     elif files_exr is not None:
#         return run_sequence_exr(files_exr, files_dots=None, start=0, stop=-1, step=1, dots=None, files_alpha=None)
    

def run_sequence_exr(files_exr, file_sunvector, files_sky_vis, file_cam_pos, files_p=None, files_dots=None, start=0, stop=-1, step=1, dots=None, use_alpha=False, specularity_danger=False):
    data = []
    res = []
    frame = start

    lst_L = walp_misc_functions.load_vector_from_file(file_sunvector, normalise=False)
    camp_pos = walp_misc_functions.load_vector_from_file(file_cam_pos, normalise=False)

    if stop == -1:
        files_exr = files_exr[start::step]
        lst_L = lst_L[start::step]
        camp_pos = camp_pos[start::step]
        if dots is not None:
            dots = dots[start::step]
        if files_dots is not None:
            files_dots = files_dots[start::step]
        if files_p is not None:
            files_dots = files_dots[start::step]
    else:
        files_exr = files_exr[start:stop:step]
        lst_L = lst_L[start:stop:step]
        camp_pos = camp_pos[start:stop:step]
        if dots is not None:
            dots = dots[start:stop:step]
        if files_dots is not None:
            files_dots = files_dots[start:stop:step]
        if files_p is not None:
            files_dots = files_dots[start:stop:step]
    pass

    if dots:
        for idx, (f_exr, L,f_sv, dot) in enumerate(zip(files_exr, lst_L, files_sky_vis, dots)):
            # TODO

            if files_p is not None:
                # We omit P as that is the real world data
                rho, _, shadow_mask, N, rgb, sky_vis = walp_misc_functions.load_data_exr(f_exr, f_sv)

                p = walp_misc_functions.load_data_p(files_p[idx], size=rgb.shape)

            else:
                rho, p, shadow_mask, N, rgb, sky_vis = walp_misc_functions.load_data_exr(f_exr, f_sv)
            
            t_data, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L)
            data.append(t_data)
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + step

    elif files_dots:
        for idx, (f_exr, L_pos, f_sv, fdot, c_pos) in enumerate(zip(files_exr, lst_L, files_sky_vis, files_dots, camp_pos)):
            # TODO
            if files_p is not None:
                rho, _, shadow_mask, N, rgb, sky_vis, dot, alpha, world_pos = walp_misc_functions.load_data_exr(f_exr, f_sv, dot_filename=fdot)

                p = walp_misc_functions.load_data_p(files_p[idx], size=rgb.shape[0:2])

            else:
                rho, p, shadow_mask, N, rgb, sky_vis, dot, alpha, world_pos = walp_misc_functions.load_data_exr(f_exr, f_sv, dot_filename=fdot)
            L_vec = L_pos / np.linalg.norm(L_pos)

            if specularity_danger:
                alpha = walp_dr.specularity_danger(N, world_pos, c_pos, L_pos, pre_mask=alpha)
            
            t_data1, t_data2, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L_vec, dot=dot, mask=alpha)
            data.append([t_data1, t_data2])
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + step

    return data, res



def run_sequence_rgb(files_normals, files_diffuse, files_shadows, files_rgb, file_sunvector, files_sky_vis, files_dots=None, start=0, stop=-1, step=1, dots=None, files_alpha=None):
    data = []
    res = []
    frame = 0

    # lst_rho, lst_p, lst_shadow_mask, lst_N, lst_L, lst_rgb = walp_misc_functions.load_data_sequence("", files_normals, files_diffuse, files_shadows, files_rgb, file_sunvector)
    # for rho, p, shadow_mask, N, L, rgb in zip(lst_rho, lst_p, lst_shadow_mask, lst_N, lst_L[150:1090:15], lst_rgb):
    lst_L = walp_misc_functions.load_light_vector(file_sunvector)

    if stop == -1:
        files_normals = files_normals[start::step]
        files_diffuse = files_diffuse[start::step]
        files_shadows = files_shadows[start::step]
        files_rgb = files_rgb[start::step]
        file_sunvector = file_sunvector[start::step]
        lst_L = lst_L[start::step]
        if dots is not None:
            dots = dots[start::step]
        if files_dots is not None:
            files_dots = files_dots[start::step]

    else:
        files_normals = files_normals[start:stop:step]
        files_diffuse = files_diffuse[start:stop:step]
        files_shadows = files_shadows[start:stop:step]
        files_rgb = files_rgb[start:stop:step]
        file_sunvector = file_sunvector[start:stop:step]
        lst_L = lst_L[start:stop:step]
        if dots is not None:
            dots = dots[start:stop:step]
        if files_dots is not None:
            files_dots = files_dots[start:stop:step]


    if files_dots is not None and files_alpha is not None:
        for f_d, f_p, f_shadow_mask, f_N, L, f_sv, f_dot, f_alpha in zip(files_diffuse, files_rgb, files_shadows, files_normals, lst_L, files_sky_vis, files_dots, files_alpha):
            rho, p, shadow_mask, N, rgb, sky_vis, dot, alpha = walp_misc_functions.load_data("", f_N, f_d, f_shadow_mask, f_p, f_sv, dot_filename=f_dot, alpha_filename=f_alpha)
            
            t_data1, t_data2, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L, dot=dot, mask=alpha)
            data.append([t_data1, t_data2])
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + 1
    elif files_dots is not None and files_alpha is None:
        for f_d, f_p, f_shadow_mask, f_N, L, f_sv, f_dot in zip(files_diffuse, files_rgb, files_shadows, files_normals, lst_L, files_sky_vis, files_dots):
            rho, p, shadow_mask, N, rgb, sky_vis, dot = walp_misc_functions.load_data("", f_N, f_d, f_shadow_mask, f_p, f_sv, dot_filename=f_dot)
            
            t_data1, t_data2, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L, dot=dot)
            data.append([t_data1, t_data2])
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + 1
    elif files_dots is None and files_alpha is not None:
        for f_d, f_p, f_shadow_mask, f_N, L, f_sv, f_alpha in zip(files_diffuse, files_rgb, files_shadows, files_normals, lst_L, files_sky_vis, files_alpha):
            rho, p, shadow_mask, N, rgb, sky_vis, alpha = walp_misc_functions.load_data("", f_N, f_d, f_shadow_mask, f_p, f_sv, alpha_filename=f_alpha)
            
            t_data1, t_data2, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L, mask=alpha)
            data.append([t_data1, t_data2])
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + 1
    elif dots is not None:
        for f_d, f_p, f_shadow_mask, f_N, L, f_sv, dot in zip(files_diffuse, files_rgb, files_shadows, files_normals, lst_L, files_sky_vis, dots):
            rho, p, shadow_mask, N, rgb, sky_vis = walp_misc_functions.load_data("", f_N, f_d, f_shadow_mask, f_p, f_sv)
            
            t_data, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L, dot=dot)
            data.append(t_data)
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + 1
    else:
        for f_d, f_p, f_shadow_mask, f_N, L, f_sv in zip(files_diffuse, files_rgb, files_shadows, files_normals, lst_L, files_sky_vis):
            rho, p, shadow_mask, N, rgb, sky_vis = walp_misc_functions.load_data("", f_N, f_d, f_shadow_mask, f_p, f_sv)
            
            t_data, t_res = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L)
            data.append(t_data)
            res.append(t_res)
            # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)

            print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *data[-1][0], *data[-1][1]))
            frame = frame + 1


    return data, res


def synth_main():
    # Run Flags
    run_pre_dot = False
    run_estimation = False
    run_validation_graph = True
    run_validation_imgs = False

    use_specularity_danger = True
    
    # Path Variables
    path_data = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\markup_street\\DS2'
    path_exr = '\\vray_raw'
    path_sky_vis = '\\SkyVis'
    path_dot_files = '\\dot_products'
    path_sun_pos = '\\sun_pos.txt'
    path_cam_pos = '\\cam_pos.txt'
    path_out_data = '\\_output\\sun_sky_data_spec_danger'
    path_out_res = '\\_output\\sun_sky_res_spec_danger'

    # Create file lists
    if run_pre_dot or run_estimation:
        files_exr, files_sky_vis, files_dots = create_input_list_exr(path_data, path_exr, path_sky_vis, path_dot_files)

    # Precalc dot products
    if run_pre_dot:
        dots = walp_misc_functions.precalc_dot(files_exr, path_data + path_sun_pos, path_data + path_dot_files, is_openexr=True, start=0, stop=-1, step=1)

    # Run Calculations
    if run_estimation:
        data, res = run_sequence_exr(files_exr, path_data + path_sun_pos, files_sky_vis, path_data + path_cam_pos, files_dots=files_dots, use_alpha=True, start=100, stop=1310, step=1, specularity_danger=use_specularity_danger)
        walp_misc_functions.save_pckl_data(path_data + path_out_data, data)
        walp_misc_functions.save_pckl_data(path_data + path_out_res, res)
   

    # Validate Results
    if run_validation_graph:
        data = np.array(walp_misc_functions.load_pckl_data(path_data + path_out_data))
        res = np.array(walp_misc_functions.load_pckl_data(path_data + path_out_res))
        # Looking until 1110 due to very dark aka bad estimates
        walp_validation.draw_sun_sky_graph(data[:,0,:], data[:,1,:], res)

    if run_validation_imgs:
        # TODO: Make it work with OpenEXR
        walp_validation.generate_data_images(files_normals, files_diffuse, files_shadows, files_rgb,
            path_data + "\\sunvector\\sun_vector.txt", files_sky_vis, data[:,0], data[:,1], files_dots, path_data + "\\out_imgs",
            start_files=100, stop_files=1360, step_files=1)

def real_main():
    # Run Flags
    run_pre_dot = False
    run_estimation = False
    run_validation_graph = False
    run_validation_imgs = True

    use_specularity_danger = True
    
    # Path Variables
    path_data = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec'
    path_p = '\\p'
    path_exr = '\\25p\\vray_raw'
    path_sky_vis = '\\sky_vis\\25p'
    path_dot_files = '\\dot_products\\25p'
    path_sun_pos = '\\sun_vec\\sun_vec.txt'
    path_cam_pos = '\\cam_vec\\cam_vec.txt'
    path_out_data = '\\_output\\sun_sky_data_spec_danger'
    path_out_res = '\\_output\\sun_sky_res_spec_danger'
    path_out_img = '\\_output\\images'

    # Create file lists
    if run_pre_dot or run_estimation or run_validation_imgs:
        files_exr, files_sky_vis, files_dots, files_p = create_input_list_exr(path_data, path_exr, path_sky_vis, path_dot_files, path_p=path_p)

    # Precalc dot products
    if run_pre_dot:
        dots = walp_misc_functions.precalc_dot(files_exr, path_data + path_sun_pos, path_data + path_dot_files, is_openexr=True, step=1)

    # Run Calculations
    if run_estimation:
        data, res = run_sequence_exr(files_exr, path_data + path_sun_pos, files_sky_vis, path_data + path_cam_pos, files_p=files_p, files_dots=files_dots, use_alpha=True, step=1, specularity_danger=use_specularity_danger)
        walp_misc_functions.save_pckl_data(path_data + path_out_data, data)
        walp_misc_functions.save_pckl_data(path_data + path_out_res, res)
   

    # Validate Results
    if run_validation_graph:
        data = np.array(walp_misc_functions.load_pckl_data(path_data + path_out_data))
        res = np.array(walp_misc_functions.load_pckl_data(path_data + path_out_res))
        # Looking until 1110 due to very dark aka bad estimates
        walp_validation.draw_sun_sky_graph(data[:,0,:], data[:,1,:], res)

    if run_validation_imgs:
        data = np.array(walp_misc_functions.load_pckl_data(path_data + path_out_data))
        # TODO: Make it work with OpenEXR
        walp_validation.generate_data_images_exr(files_exr,
            path_data + path_sun_pos, files_sky_vis, data[:,0], data[:,1], files_dots, path_data + path_out_img,
            )
        # walp_validation.generate_data_images(files_normals, files_diffuse, files_shadows, files_rgb,
        #     path_data + path_sun_pos, files_sky_vis, data[:,0], data[:,1], files_dots, path_data + path_out_img,
        #     start_files=100, stop_files=1360, step_files=1)
    

if __name__ == "__main__":
    # synth_main()
    real_main()