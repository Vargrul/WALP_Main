import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import pairwise as pw
from scipy import ndimage

def __create_mask_1d_to_3d(mask_1, mask_2 = None):
    mask = None
    if type(mask_2) != None:
        # Make 3 dimensional
        mask = np.dstack((np.dstack((np.logical_and(mask_1, mask_2),np.logical_and(mask_1, mask_2))),np.logical_and(mask_1, mask_2)))
    else:
        mask = np.dstack((np.dstack((mask_1, mask_1)), mask_1))

    return mask

def calc_area_mean_rgb(rgb_img, area_mask, alpha_mask = None):
    mask = __create_mask_1d_to_3d(area_mask, alpha_mask)
    masked_array = np.ma.array(rgb_img, mask=mask)

    return np.array(masked_array.mean(axis=(0, 1)))

def calc_sun_shadow_factor(img_light_contrib, groups, mask_shadow, mask_sun, mask_alpha = None):
    # Create masks
    if type(mask_alpha) != None:
        mask_shadow = np.logical_and(mask_shadow, mask_alpha)
        mask_sun = np.logical_and(mask_sun, mask_alpha)

    group_values = []
    sum_pos = sum([len(g) for g in groups])
    for i, g in enumerate(groups):
        mask_group = np.zeros(mask_shadow.shape)
        weight_group = len(g) / sum_pos

        for pos in g:
            mask_group[pos[0], pos[1]] = True

        mask_temp_1 = __create_mask_1d_to_3d(np.logical_and(mask_group, mask_shadow), mask_alpha)
        # sum_test = __calc_mean_mask(img_light_contrib, mask_temp_1)
        # print(sum_test)
        ma_shadow = np.ma.array(img_light_contrib, mask=mask_temp_1)
        val_shadow = ma_shadow.mean(dtype=np.float32)

        mask_temp_2 = __create_mask_1d_to_3d(np.logical_and(mask_group, mask_sun), mask_alpha)
        ma_sun = np.ma.array(img_light_contrib, mask=mask_temp_2)
        val_sun = ma_sun.mean(dtype=np.float32)

        factor = val_shadow / val_sun

        print("Group: {}\tWeight: {}\tShadow Value: {}\tSun Value: {}\tFactor: {}\tMax: {},{}\tMin: {},{}".format(i, weight_group, val_shadow, val_sun, factor, ma_shadow.max(), ma_sun.max(), ma_shadow.min(), ma_sun.min()))

def __calc_mean_mask(img, mask):
    if np.all(False):
        return 0
    count = 0
    pos_sum = 0
    for x, col in enumerate(img):
        for y, pos in enumerate(col):
            # print(mask[x,y])
            if mask[x,y,0]:
                count = count + 1            
                pos_sum = pos_sum + sum([i for i in pos])

    return pos_sum/count

def cos_similar_normals(normals, mask_alpha, sim_const = 0.90):
    # Create mask
    mask = np.copy(mask_alpha)

    normals_norm = np.linalg.norm(normals, axis=2)

    # Compare normals to each other
    similar_normals = []
    for ax, ac in enumerate(normals_norm):
        for ay, a in enumerate(ac):
            if not mask[ax,ay]:
                continue
            
            print("Claculating for point {},{}".format(ax, ay))

            # Compare A to all vectors
            cur_norm_similar = []
            for bx, bc in enumerate(normals_norm):
                for by, b in enumerate(bc):
                    if not mask[bx,by]:
                        continue
                    cos_sim = np.dot(normal_vectors[ax][ay], normal_vectors[bx][by]) / ( a*b )
                    if cos_sim > sim_const:
                        cur_norm_similar.append([bx,by])
                        mask[bx,by] = False
            
            similar_normals.append(cur_norm_similar)
            print("Found {} matches".format(len(cur_norm_similar)))
    
    return similar_normals
    # print(pw.cosine_similarity(normal_vectors[592, 418], normal_vectors[427, 1129]))
    
def __img_from_similarity_gorups(arr_shape, sim_groups):
    group_img = np.zeros(arr_shape)
    for i, g in enumerate(sim_groups):
        for n in g:
            group_img[n[0],n[1]] = i

    return group_img
        


if __name__ == "__main__":
    # Load all images and masks
    shadow_img = np.array(Image.open('../data/markup_street/markup_street.VRayShadows.jpg').convert('L'))
    shadow_mask = shadow_img > 1
    sun_mask = ~shadow_mask
    shadow_mask = ndimage.binary_erosion(shadow_mask, structure=np.ones((2,2))).astype(shadow_mask.dtype)
    sun_mask = ndimage.binary_erosion(sun_mask, structure=np.ones((2,2))).astype(sun_mask.dtype)

    rgb_img = np.array(Image.open('../data/markup_street/markup_street.RGB_color.jpg'))
    
    alpha_img = np.array(Image.open('../data/markup_street/markup_street.Alpha.jpg').convert('L'))
    alpha_mask = alpha_img > 1
    alpha_mask = ndimage.binary_erosion(alpha_mask, structure=np.ones((2,2))).astype(alpha_mask.dtype)

    diffuse_reflectance_img = np.array(Image.open('../data/markup_street/markup_street.VRayDiffuseFilter.jpg'))

    normals_img = np.array(Image.open('../data/markup_street/markup_street.VRayBumpNormals.jpg'))
    normals_img = normals_img/255.0

    normal_vectors = normals_img * 2 - 1


    # Calculate shadow/light area factor
    mean_pixel_sun = calc_area_mean_rgb(rgb_img, shadow_mask, alpha_mask=alpha_mask)
    mean_pixel_shadow = calc_area_mean_rgb(rgb_img, ~shadow_mask, alpha_mask=alpha_mask)
    light_shadow_factor = mean_pixel_sun/mean_pixel_shadow
    print(
        'Mean Pixel Value light area: {}\tMean Pixel Value shadow area: {}'.format(mean_pixel_sun, mean_pixel_shadow)
        )
    print(
        'Factor between: {}'.format(light_shadow_factor)
        )

    
    # Calculate the lightcontribution from reflectance
    light_contrib_img = rgb_img/diffuse_reflectance_img
    light_contrib_img[light_contrib_img==np.inf] = 0

    light_contrib_img[~alpha_mask] = 0

    print(np.amax(light_contrib_img))
    # light_contrib_img[light_contrib_img>100] = 0
    light_contrib_img_normalized = light_contrib_img*(1.0/np.amax(light_contrib_img))

    mean_light_sun = calc_area_mean_rgb(light_contrib_img, shadow_mask, alpha_mask=alpha_mask)
    mean_light_shadow = calc_area_mean_rgb(light_contrib_img, sun_mask, alpha_mask=alpha_mask)
    light_sun_shadow_factor = mean_light_sun/mean_light_shadow
    print(
        'Mean sun color: {}\tMean shadow color: {}'.format(mean_light_sun, mean_light_shadow)
        )
    print(
        'Factor between: {}'.format(light_sun_shadow_factor)
        )

    # TODO solve the cos -> lightsource problem
    # -This could be via using the known sun direction or
    # -sampling similar normal direction points.
    # Normal direction sampling could be done by taking the mean normal directions and sample similar to that.
    norm_groups = cos_similar_normals(normal_vectors, alpha_mask)
    norm_groups_img = __img_from_similarity_gorups(shadow_mask.shape, norm_groups)

    calc_sun_shadow_factor(light_contrib_img_normalized, norm_groups, shadow_mask, sun_mask, alpha_mask)

    plot = plt.subplot(2,4,1).imshow(np.logical_and(shadow_mask, alpha_mask), cmap='gray')
    plot = plt.subplot(2,4,2).imshow(np.logical_and(~shadow_mask, alpha_mask), cmap='gray')
    plot = plt.subplot(2,4,3).imshow(norm_groups_img)
    plot = plt.subplot(2,4,5).imshow(rgb_img)
    plot = plt.subplot(2,4,6).imshow(diffuse_reflectance_img)
    plot = plt.subplot(2,4,7).imshow(light_contrib_img_normalized)
    plot = plt.subplot(2,4,8).imshow(normal_vectors)
    plt.show()
    # plot = plt.imshow(alpha_mask, cmap=plt.get_cmap('gray'))
    # plt.show()

