import pandas as pd
import numpy as np
import glob

# Read processed excel file
xlsx_dir = "H:/data/brc/Duke-Breast-Cancer-MRI/Clinical_and_Other_Features_Recurrence.xlsx"
data = pd.read_excel(xlsx_dir) # "openpyxl" library needs to be installed for "read_excel" function
data.replace(['NP', 'NC', ''], np.nan, inplace=True)

# Add a new 'RECURwithin3yrs' column and fill in 'np.nan' values
data['RECURwithin3yrs'] = np.nan

for i in range(len(data)):
    if data['Days to Earlier recurrence'][i] <= 1095:
        data['RECURwithin3yrs'][i] = 'RECURwithin3yrs'

    elif data['Days to Earlier recurrence'][i] >= 1095 & data['Days to Earlier recurrence free'][i] >= 1095:
        data['RECURwithin3yrs'][i] = 'RECURafter3yrs'

    elif data['Days to Earlier recurrence'][i] == np.nan & data['Days to Earlier recurrence free'][i] >= 1095:
        data['RECURwithin3yrs'][i] = 'NoRECUR'  # NoRECURwithin3yrs

    elif data['Days to Earlier recurrence'][i] == np.nan & data['Days to Earlier recurrence free'][i] <= 1095:
        data['RECURwithin3yrs'][i] = 'NotSure'

data.to_excel(
    "H:/data/brc/Duke-Breast-Cancer-MRI/Clinical_and_Other_Features_Recurrence_Processed.xlsx")

# Make Metadata

resized_NIFTI_dir = "H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/1.resized_files/96_96_32/"
resized_NIFTI_list = glob.glob(resized_NIFTI_dir +'*/*.nii')
resized_NIFTI_list.sort()

data = pd.DataFrame(resized_NIFTI_list)
data.columns = ['NiftiFilePath'] # change the column name to 'NiftiFilePath'
data['OriginatedDataset'] = "Duke-Breast-Cancer-MRI"
data['Patient_ID'] = None
data['NiftiFileName'] = None

for i in range(len(data)):
    Patient_ID_Index = data['NiftiFilePath'][i].find("Breast_MRI_")
    data['Patient_ID'][i] = data['NiftiFilePath'][i][Patient_ID_Index : Patient_ID_Index + 14]
    NiftiFileName_Index = data['NiftiFilePath'][i].rfind("\\")
    data['NiftiFileName'][i] = data['NiftiFilePath'][i][NiftiFileName_Index+1 : ]
    
Clinical_Features = pd.read_excel("H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_Metadata/Clinical_and_Other_Features.xlsx")

for i in Clinical_Features.columns[2:]:
    data[i] = None
    for j in range(len(data)):
        Index = Clinical_Features[Clinical_Features['Patient_ID'] == data['Patient_ID'][j]].index[0]
        data[i][j] = Clinical_Features[i][Index]
        
dir = "H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_Metadata/"
file_name = "MetadataDuke.xlsx"
data.to_excel(dir + file_name, index_label= 'Index')

# Make labels

image_nparrays = np.load("H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/2.normalized_nparrrays/96_96_32/96_96_32_min-max-normalized_float32.npy")
MetadataDuke = pd.read_excel("H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_Metadata/MetadataDuke.xlsx")
dic = {'NoRECUR' : 0, 'RECURwithin3yrs' : 1, 'RECURafter3yrs' : 2, 'NotSure' : 'NA'}
labels = np.array([dic[j] for j in MetadataDuke['RECURwithin3yrs']])
unique_labels = np.unique(labels)

nparrays = {}
for j in unique_labels:
    indices = np.isin(labels, j)
    nparrays[j] = image_nparrays[indices]
    print(f"the shape of {j} numpy array is {nparrays[j].shape}")

Available_indices = labels != 'NA'
Available_nparray = image_nparrays[Available_indices]
Available_labels = labels[Available_indices]

print(f"the shape of Available numpy array is {Available_nparray.shape}")
print(f"the shape of Available labels is {Available_labels.shape}")

np.save('H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/2.normalized_nparrrays/Available_images.npy', Available_nparray)
np.save('H:/data/brc/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI_NIFTI/2.normalized_nparrrays/Available_labels.npy', Available_labels)