# DICOM to NIFTI

import glob
import os
import dicom2nifti
import nibabel as nib
import skimage.transform as skTrans
import numpy as np

os.chdir('H:/data/brc/Duke-Breast-Cancer-MRI')
dataset_dir = 'H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_DICOM/'


def dicom_to_nifti(dataset_dir):
    DICOM_dirlist = glob.glob(os.path.join(dataset_dir, '*/*/*'))
    DICOM_dirlist.sort()
    for i in DICOM_dirlist:
        nii_dir = os.path.join(
            dataset_dir[:35], "Duke-Breast-Cancer-MRI_NIFTI/0.original_files", i[64:79])
        os.makedirs(nii_dir, exist_ok=True)
        dicom2nifti.convert_directory(i, nii_dir, compression=False)
    print("All DICOM files are compiled to NIFTI files")
    print("Please refer to", os.path.join(
        dataset_dir[:35], "Duke-Breast-Cancer-MRI_NIFTI/0.original_files/"))


def inspect(nii_dir):
    nii_files = sorted(glob.glob(nii_dir + '*/*.nii'))
    nii_files_shape_dict = {}
    nii_files_orientation_dict = {}

    for i in nii_files:
        img_instance = nib.load(i)
        img_array = img_instance.get_fdata()
        img_shape = img_array.shape
        nii_files_shape_dict[img_shape] = nii_files_shape_dict.get(
            img_shape, 0) + 1

        orientation = nib.aff2axcodes(img_instance.affine)
        nii_files_orientation_dict[orientation] = nii_files_orientation_dict.get(
            orientation, 0) + 1

    with open('inspect.txt', 'w') as data:
        for orientation, count in nii_files_orientation_dict.items():
            data.write("{}\t{}\n".format(orientation, count))
        for shape, count in nii_files_shape_dict.items():
            data.write("{}\t{}\n".format(shape, count))

# nii_dir = 'H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/0.original_files/'
# resized_dir = 'H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/1.resized_files/96_96_32/'
# target_shape = (96, 96, 32)


def resize(resized_dir, nii_dir, target_shape):
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)

    nii_files = sorted(glob.glob(nii_dir + '*/*.nii'))

    for i in nii_files:
        sub_dir = os.path.join(resized_dir, i[i.index(
            'Breast_MRI_'):i.index('Breast_MRI_')+15])
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        img = nib.load(i).get_fdata()
        img_resized = skTrans.resize(
            img, target_shape, order=1, preserve_range=True)
        img_nparray = np.array(img_resized, dtype=np.float32)
        img_nifti = nib.Nifti1Image(img_nparray, affine=np.eye(4))
        nib.save(img_nifti, os.path.join(sub_dir, i[i.index('Breast_MRI_'):]))
        print(f"{i[i.index('Breast_MRI_'):]} is processed.")

    print(f"All {len(nii_files)} NIFTI files are resized and saved.")

# resized_dir = 'H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/1.resized_files/96_96_32/'
# save_dir = 'H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/2.normalized_nparrrays/96_96_32/'
# file_name = '96_96_32_min-max-normalized_float32.npy'


def minmaxnorm(resized_dir, save_dir, file_name):
    resized_files = glob.glob(resized_dir+'*/*.nii')
    resized_files.sort()
    images = [nib.load(f) for f in resized_files]
    image_nparrays = nib.funcs.concat_images(images).get_fdata()
    image_nparrays = (image_nparrays - np.min(image_nparrays)
                      ) / np.max(image_nparrays)
    image_nparrays = image_nparrays.astype(np.float32)

    np.save(os.path.join(save_dir, file_name), image_nparrays)
    print(image_nparrays.shape)
