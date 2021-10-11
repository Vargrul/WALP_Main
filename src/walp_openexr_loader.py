from numpy import append, mask_indices
import OpenImageIO as oiio
import numpy as np
from PIL import Image
import time

def get_img_specs(path):
    img_file = oiio.ImageInput.open(path)
    spec = img_file.spec()
    img_file.close()
    return spec

def get_all_channels(path):
    img_file = oiio.ImageInput.open(path)
    img = img_file.read_image()
    img_file.close()
    return img

def get_img_channel(path, image_channel_name, is_rgb=None):
    if type(image_channel_name) is not tuple and type(image_channel_name) is not list:
        raise TypeError("Expected a list or a tuple")

    if is_rgb is None:
        is_rgb = [True for _ in image_channel_name]
    else:
        if len(is_rgb) is not len(image_channel_name):
            raise ValueError("Amoung of image channels and rgb channel booleans needs to be equal")

    tries = 20
    while tries > 0:
        img_file = oiio.ImageInput.open(path)
        if img_file is not None:
            break
        tries = tries - 1
        print("Retrying to load image")
        time.sleep(0.01)
        if tries == 0:
            print('Could not open "' + path + '"')
            print("\tError: ", oiio.geterror())

    img_spec = img_file.spec()
    img = img_file.read_image()
    
    ret_data = []
    for chn, i_rbg in zip(image_channel_name, is_rgb):
        first_channel_index = None
        for idx, ch in enumerate(img_spec.channelnames):
            if ch.startswith(chn):
                first_channel_index = idx
                break
        
        if i_rbg:
            ret_data.append(img[:,:,first_channel_index:first_channel_index+3])
        else:
            ret_data.append(img[:,:,first_channel_index])
    
    img_file.close()
    
    return ret_data


def main():
    file_path = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\markup_street\\DS5\\vray_raw\\'
    file_name = 'top_scene.1079.exr'
    img = get_img_channel(file_path + file_name, ['', 'VRaySampleRate'])
    uint_img = np.uint8(np.minimum(255,img[0]/0.3*255))
    pil_img = Image.fromarray(uint_img)
    pil_img.show()
    uint_img = np.uint8(np.minimum(255,img[1]/img[1].max()*255))
    pil_img = Image.fromarray(uint_img)
    pil_img.show()

if __name__ == "__main__":
    main()