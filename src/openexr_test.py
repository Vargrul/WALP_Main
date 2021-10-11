import OpenImageIO as oiio

print(oiio.__version__)


file_path = 'E:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\markup_street\\test_new_renderelement\\'
img_input = oiio.ImageInput.open(file_path + 'top_scene.1079.exr')
img_spec = img_input.spec()
img = img_input.read_image()

for idx, ch in enumerate(img_spec.channelnames):
    if ch.startswith('VRayWorldPoint'):
        print(idx)

print(img_spec.channelnames.index('VRayWorldPoint.R'))

print ("extra_attribs size is", len(img_spec.extra_attribs))
for i in range(len(img_spec.extra_attribs)) :
    print (i, img_spec.extra_attribs[i].name, str(img_spec.extra_attribs[i].type), " :")
    print ("\t", img_spec.extra_attribs[i].value)
print