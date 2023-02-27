import os
import pandas as pd
import nibabel as nib
import numpy as np
import torchio as tio
from sklearn.model_selection import StratifiedShuffleSplit
import glob

# Data Augmentation


def trainset_aug(NumpyPath, MetadataPath, AugFileSaveDir, AugMetadataSaveDir):
    df = pd.read_excel(MetadataPath, header=0, index_col=0)
    df = df.query('RECURwithin3yrs != "NotSure"')
    df['RECURwithin3yrs'].replace(
        {'NoRECUR': 0, 'RECURwithin3yrs': 1, 'RECURafter3yrs': 2}, inplace=True)

    NiftiFilePath = df.loc[:, 'NiftiFilePath'].values
    Patient_ID = df.loc[:, 'Patient_ID'].values
    NiftiFileName = df.loc[:, 'NiftiFileName'].values
    Age = df.loc[:, 'Age'].values
    ER = df.loc[:, 'ER'].values
    PR = df.loc[:, 'PR'].values
    HER2 = df.loc[:, 'HER2'].values
    RECURwithin3yrs = df.loc[:, 'RECURwithin3yrs'].values

    x = np.load(NumpyPath)
    y = np.array(RECURwithin3yrs)
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=2022)
    train_idx, test_idx = next(split.split(x, y))

    x_train = x[train_idx]
    y_train = y[train_idx]
    NiftiFilePath_train = NiftiFilePath[train_idx]
    Patient_ID_train = Patient_ID[train_idx]
    NiftiFileName_train = NiftiFileName[train_idx]
    Age_train = Age[train_idx]
    ER_train = ER[train_idx]
    PR_train = PR[train_idx]
    HER2_train = HER2[train_idx]
    RECURwithin3yrs_train = RECURwithin3yrs[train_idx]

    target_shape = (1, x_train.shape[1], x_train.shape[2], x_train.shape[3])
    transforms_dict0 = {tio.RandomNoise(): 0.3}
    transforms_dict1 = {tio.RandomBiasField(): 0.3}
    transforms_dict2 = {tio.RandomBlur(): 0.3}
    transforms_dict3 = {tio.RandomGamma(log_gamma=(-2, 2)): 0.1}
    transforms_dict = [transforms_dict0, transforms_dict1,
                       transforms_dict2, transforms_dict3]

    with open(AugMetadataSaveDir, 'w') as fHnd:
        for index, (file_path, patient_id, age, er, pr, her2, recur_within_3yrs) in enumerate(zip(NiftiFilePath_train, Patient_ID_train, Age_train, ER_train, PR_train, HER2_train, RECURwithin3yrs_train)):
            img_array = nib.load(file_path).get_fdata()
            converted_array = np.array(img_array, dtype=np.float32)
            converted_array = converted_array.reshape(target_shape)

            for j in range(4):
                transform = tio.OneOf(transforms_dict[j])
                img_transform = transform(converted_array)
                NiftiAugFileName = NiftiFileName_train[index].replace(
                    ".nii", ".aug") + str(j) + ".nii"
                nib.save(nib.Nifti1Image(img_transform, affine=np.eye(4)),
                         AugFileSaveDir + patient_id + "/" + NiftiAugFileName)

                fHnd.write(str(index) + "\t" +
                           AugFileSaveDir + Patient_ID_train[index] + "/" + NiftiAugFileName + "\t" +
                           Patient_ID_train[index] + "\t" +
                           NiftiAugFileName + "\t" +
                           str(age) + "\t" +
                           str(er) + "\t" +
                           str(pr) + "\t" +
                           str(her2) + "\t" +
                           str(recur_within_3yrs) + "\n")
            print(patient_id, NiftiFileName_train[index], "was Augmented.")

        fHnd.close()

# NumpyPath = "/data/Projects/Deep_learning/eg/data/NIFTI/2.normalized_nparrays&labels/96_96_32/Available_images.npy"
# MetadataPath = "/data/Projects/Deep_learning/eg/data/Metadata/MetadataDuke_96_96_32_DeeplearningServer.xlsx"
# AugFileSaveDir = "/data/Projects/Deep_learning/eg/data/NIFTI/3.augmented/96_96_32/train_test_split_random_state_2022/"
# AugMetadataSaveDir = "/data/Projects/Deep_learning/eg/data/Metadata/96_96_32_NIFTI_Aug.txt"
# trainset_aug(NumpyPath, MetadataPath, AugFileSaveDir, AugMetadataSaveDir)


# Augmented Data Normalization & Make Numpy array
df = pd.read_csv("/data/Projects/Deep_learning/eg/data/Metadata/96_96_32_NIFTI_Aug.txt", sep="\t", encoding="utf-8", index_col=0,
                 header=None, names=["index",
                                     "NiftiFilePath",
                                     "Patient_ID",
                                     "NiftiFileName",
                                     "Age",
                                     "ER",
                                     "PR",
                                     "HER2",
                                     "RECURwithin3yrs"])

NiftiFilePath_list = list(df["NiftiFilePath"])
Patient_ID_list = list(df["Patient_ID"])
NiftiFileName_list = list(df["NiftiFileName"])
Age_list = list(df["Age"])
ER_list = list(df["ER"])
PR_list = list(df["PR"])
HER2_list = list(df["HER2"])
RECURwithin3yrs_list = list(df["RECURwithin3yrs"])


def nparray_aug(AugFileSaveDir, NumpyarraySaveDir, ImageName, LabelName):
    Images_npy = []
    Labels_npy = []
    for i in range(len(NiftiFilePath_list)):
        img = nib.load(NiftiFilePath_list[i]).get_fdata()
        img = (img - np.min(img))/np.max(img)
        img = img.astype(np.float32)
        img = img.reshape(96, 96, 32)
        Images_npy.append(img)
        Labels_npy.append(RECURwithin3yrs_list[i])

    Images_npy = np.array(Images_npy)
    Labels_npy = np.array(Labels_npy)

    np.save(NumpyarraySaveDir + ImageName, Images_npy)
    np.save(NumpyarraySaveDir + LabelName, Labels_npy)

# AugFileSaveDir = '/data/Projects/Deep_learning/eg/data/NIFTI/3.augmented/96_96_32/train_test_split_random_state_2022/'
# NumpyarraySaveDir = '/data/Projects/Deep_learning/eg/data/NIFTI/3.augmented/96_96_32/'
# ImageName = 'Augmented_96_96_32_min-max-normalized_float32_images.npy'
# LabelName = 'Augmented_96_96_32_min-max-normalized_float32_labels.npy'
# nparray_aug(AugFileSaveDir, NumpyarraySaveDir, ImageName, LabelName)
