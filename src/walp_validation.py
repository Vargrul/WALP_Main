import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import walp_re_render
import walp_misc_functions
import walp_light_solver_lstsq

def validate_data(gt_img, sun, sky, rho, sm, c, N, L):
    # Render using the assumed model
    sim_img = walp_re_render.sim_data(rho, sun, sky, sm, c, gt_img.shape, N=N, L=L)
    
    
    # Calculate validation parameters
        # Difference Image
    dif_img = np.abs(np.subtract(sim_img.astype(np.float), gt_img.astype(np.float))).astype(np.uint8)
        # STD
        # Mean Error
        # % difference (How is this calculcated?)
    # Prepare a vector of the Light Vector for the dot product calculations
    L_ar = np.ones(N.shape) * L

    # Calculate the dot product for each normal in regards to the light vector
    dot = walp_misc_functions.mp_dot(N, L_ar)

    plot = plt.subplot(1,3,1).imshow(gt_img)
    plot = plt.subplot(1,3,2).imshow(sim_img)
    plot = plt.subplot(1,3,3).imshow(np.reshape(walp_light_solver_lstsq.a(rho[:,0], sm, N=N, L=L), (gt_img.shape[0], gt_img.shape[1])))
    # plot = plt.subplot(1,3,3).imshow(dif_img)
    plt.show()

def draw_sun_sky_graph(sun, sky, res):
    # Calculate sum and plot
    sum_pow = np.sum([sun,sky], axis=0)
    # print(sum_pow)
    
    plt.subplot(1,2,1).plot(sun)
    plt.subplot(1,2,1).plot(sky)
    plt.subplot(1,2,1).plot(sum_pow)
    plt.subplot(1,2,1).legend(["Sun R", "Sun G", "Sun B", "Sky R", "Sky G", "Sky B", "Sum R", "Sum G", "Sum B"])
    plt.subplot(1,2,1).grid()

    plt.subplot(1,2,2).plot(res[:,0])
    plt.subplot(1,2,2).plot(res[:,1])
    plt.subplot(1,2,2).plot(res[:,2])
    plt.subplot(1,2,2).legend(["Residual R", "Residual G", "Residual B"])
    plt.subplot(1,2,2).grid()

    plt.show()
    pass

def draw_residual_graph(res):
    # print(sum_pow)
    
    plt.plot(res)
    plt.legend(["residual"])
    plt.grid()
    plt.show()

def generate_data_images_exr(files_exr, file_sunvector, files_sky_vis, data_Is, data_Ia, files_dots, path_out,
    start_files=0, stop_files=-1, step_files=1, start_data=0, stop_data=-1, step_data=1):
    frame = 0

    
    lst_L = walp_misc_functions.load_vector_from_file(file_sunvector, normalise=False)

    if stop_files == -1:
        files_exr = files_exr[start_files::step_files]
        file_sunvector = file_sunvector[start_files::step_files]
        lst_L = lst_L[start_files::step_files]
        files_dots = files_dots[start_files::step_files]
    else:
        files_exr = files_exr[start_files:stop_files:step_files]
        file_sunvector = file_sunvector[start_files:stop_files:step_files]
        lst_L = lst_L[start_files:stop_files:step_files]
        files_dots = files_dots[start_files:stop_files:step_files]

    if stop_data == -1:
        data_Is = data_Is[start_data::step_data]
        data_Ia = data_Ia[start_data::step_data]
    else:
        data_Is = data_Is[start_data:stop_data:step_data]
        data_Ia = data_Ia[start_data:stop_data:step_data]

    for f_exr, L, f_sv, f_dot, Is, Ia in zip(files_exr, lst_L, files_sky_vis, files_dots, data_Is, data_Ia):
        rho, _, sm, N, rgb, c, dot, alpha, world_pos = walp_misc_functions.load_data_exr(f_exr, f_sv, dot_filename=f_dot)

        generate_img(rho, Is, Ia, sm, c, rgb, dot, path_out, frame)
        
        print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *Is, *Ia))
        frame = frame + 1
    pass

def generate_data_images(files_normals, files_diffuse, files_shadows, files_rgb, file_sunvector, files_sky_vis,
    data_Is, data_Ia, files_dots, path_out,
    start_files=0, stop_files=-1, step_files=1, start_data=0, stop_data=-1, step_data=1):
    frame = 0

    # lst_rho, lst_p, lst_shadow_mask, lst_N, lst_L, lst_rgb = walp_misc_functions.load_data_sequence("", files_normals, files_diffuse, files_shadows, files_rgb, file_sunvector)
    # for rho, p, shadow_mask, N, L, rgb in zip(lst_rho, lst_p, lst_shadow_mask, lst_N, lst_L[150:1090:15], lst_rgb):
    lst_L = walp_misc_functions.load_light_vector(file_sunvector)

    if stop_files == -1:
        files_normals = files_normals[start_files::step_files]
        files_diffuse = files_diffuse[start_files::step_files]
        files_shadows = files_shadows[start_files::step_files]
        files_rgb = files_rgb[start_files::step_files]
        file_sunvector = file_sunvector[start_files::step_files]
        lst_L = lst_L[start_files::step_files]
        files_dots = files_dots[start_files::step_files]
    else:
        files_normals = files_normals[start_files:stop_files:step_files]
        files_diffuse = files_diffuse[start_files:stop_files:step_files]
        files_shadows = files_shadows[start_files:stop_files:step_files]
        files_rgb = files_rgb[start_files:stop_files:step_files]
        file_sunvector = file_sunvector[start_files:stop_files:step_files]
        lst_L = lst_L[start_files:stop_files:step_files]
        files_dots = files_dots[start_files:stop_files:step_files]

    if stop_data == -1:
        data_Is = data_Is[start_data::step_data]
        data_Ia = data_Ia[start_data::step_data]
    else:
        data_Is = data_Is[start_data:stop_data:step_data]
        data_Ia = data_Ia[start_data:stop_data:step_data]
            
    for f_d, f_p, f_shadow_mask, f_N, L, f_sv, f_dot, Is, Ia in zip(files_diffuse, files_rgb, files_shadows, files_normals, lst_L, files_sky_vis, files_dots, data_Is, data_Ia):

        # rho, _, sm, N, rgb, c, dot, alpha, world_pos = walp_misc_functions.load_data_exr(f_exr, f_sv, dot_filename=fdot)

        rho, p, sm, N, rgb, c, dot, _ = walp_misc_functions.load_data("", f_N, f_d, f_shadow_mask, f_p, f_sv, dot_filename=f_dot)
        

        generate_img(rho, Is, Ia, sm, c, rgb, dot, path_out, frame)
        # dot = np.einsum('ij, ij->i', N, L_ar)
        # dot = np.reshape(dot, (dot.shape[0],1))

        # data.append(walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, sky_vis, N, L, dot=dot))
        # sim_img = walp_re_render.sim_data(rho, Is, Ia, sm, c, rgb.shape, dot=dot)
        # # plt.imshow(sim_img)
        # # plt.show()
        # sim_im = Image.fromarray(sim_img)
        # sim_im.save(path_out + "\\sim_imgs\\sim_img_" + "{:05}.jpeg".format(frame))
        # # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)
        
        # dif_img = np.abs(np.subtract(sim_img.astype(np.float), rgb.astype(np.float)*255)).astype(np.uint8)
        # dif_im = Image.fromarray(dif_img)
        # dif_im.save(path_out + "\\dif_imgs\\dif_img_" + "{:05}.jpeg".format(frame))

        print("Frame: {:4}\tSun: {:10.3f}{:10.3f}{:10.3f}\tSky: {:10.3f}{:10.3f}{:10.3f}".format(frame, *Is, *Ia))
        frame = frame + 1

def generate_img(rho, Is, Ia, sm, c, rgb, dot, path_out, frame):
    sim_img = walp_re_render.sim_data(rho, Is, Ia, sm, c, rgb.shape, dot=dot)
    # plt.imshow(sim_img)
    # plt.show()
    sim_im = Image.fromarray(sim_img)
    sim_im.save(path_out + "\\sim_imgs\\sim_img_" + "{:05}.jpeg".format(frame))
    # sun, sky = walp_light_solver_lstsq.est_lstsq_rgb(rho, p, shadow_mask, N, L)
    
    dif_img = np.abs(np.subtract(sim_img.astype(np.float), rgb.astype(np.float)*255)).astype(np.uint8)
    dif_im = Image.fromarray(dif_img)
    dif_im.save(path_out + "\\dif_imgs\\dif_img_" + "{:05}.jpeg".format(frame))