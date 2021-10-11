import pickle
import numpy as np

def load_pckl_data(name):
    if name[-4:] == ".pkl":
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    path_data = 'D:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec'
    path_out_data = '\\_output\\sun_sky_data_spec_danger'
    path_out_res = '\\_output\\sun_sky_res_spec_danger'

    data = load_pckl_data(path_data + path_out_data)

    sun = np.array(data[0][0])
    sky = np.array(data[0][1])

    sun_color = min(1/sun)*sun*255
    sky_color = min(1/sky)*sky*255
    sun_intensity = sum(sun)/3
    sky_intensity = sum(sky)/3

    pass