import json
import numpy as np
import os.path as path
import nibabel as nib
import os
import glob
# import ipdb
import SimpleITK as sitk
import pydicom
from skimage.transform import resize
from tqdm import tqdm
import json



FLARE_path = "/data/MING/data/FLARE/train/images"
FLARE_label_path = "/data/MING/data/FLARE/train/labels"
flare_to_path = "/data/MING/data/FLARE/new/image"
flare_label_to_path = "/data/MING/data/FLARE/new/label"
if not os.path.exists(flare_to_path):
	os.makedirs(flare_to_path)
if not os.path.exists(flare_label_to_path):
	os.makedirs(flare_label_to_path)

names = glob.glob(path.join(FLARE_path, '*.gz'))
label_names = glob.glob(path.join(FLARE_label_path, '*.gz'))
label_names.sort()

names.sort()

names = [path.split(f)[1] for f in names]
label_names = [path.split(f)[1] for f in label_names]

pad = [32,32,32]
for i in tqdm(range(len(names))):

	img_name = names[i]


	label_name = label_names[i]
	print(img_name)
	print(label_name)

	image = nib.load(path.join(FLARE_path, img_name))
	spacing = image.affine[[0,1,2], [0,1,2]]

	ind = ((-spacing>0)-0.5)*2
	image = image.get_fdata()
	image = np.transpose(image,[1,0,2])
	image = image[::int(ind[1]),::int(ind[0]),::int(ind[2])]
	aug_spacing = np.array([1.5, 1.5, 2.0])
	new_size = (np.array(image.shape)*np.abs(spacing)).astype(np.int)
	new_size = (new_size / np.abs(aug_spacing)).astype(np.int)

	image = resize(image.astype(np.float64), new_size)

	label = nib.load(path.join(FLARE_label_path, label_name))
	spacing = label.affine[[0,1,2],[0,1,2]]
	label = label.get_fdata()

	label = np.transpose(label,[1,0,2])
	ind = ((-spacing>0)-0.5)*2
	label = label[::int(ind[1]),::int(ind[0]),::int(ind[2])]
	label = resize(label.astype(np.float64),new_size,anti_aliasing=False,order=0)
	print(img_name, 'loaded', new_size, spacing)
	print("image shape {}".format(image.shape))
	print("label shape {}".format(label.shape))

	Out_img = image
	Out_label = label


	path_prefix = path.join(flare_to_path, img_name.split('.')[0]) + ".nii.gz"

	label_path_prefix = path.join(flare_label_to_path, label_name.split('.')[0]) + ".nii.gz"

	nii_file = np.swapaxes(Out_img, 0, 2)

	nii_file = sitk.GetImageFromArray(nii_file)
	sitk.WriteImage(nii_file, path_prefix)

	nii_file = np.swapaxes(Out_label, 0, 2)
	nii_file = sitk.GetImageFromArray(nii_file)
	sitk.WriteImage(nii_file, label_path_prefix)



def save_json(flare_path, flare_label_path, save_path):
    flare_labels = glob.glob(os.path.join(flare_label_path, "*.nii.gz"))
    flare_labels.sort()
    flare = glob.glob(os.path.join(flare_path,"*.nii.gz"))
    flare.sort()
    pcr_json = {"FLARE_train": [], "FLARE_val": []}
    for i in range(len(flare)):
        file = flare[i]
        label_file = flare_labels[i]
        pcr_json["FLARE_val"].append({"image": file, "label": label_file})
    json_str = json.dumps(pcr_json, indent=4)
    with open(os.path.join(save_path,"flare.json"), 'w') as json_file:
        json_file.write(json_str)

