import os
import xml.etree.ElementTree as ET
from datetime import datetime
cwd = os.getcwd()
print("current directory is: ", cwd)

fileNAME = os.path.join(cwd, "dataset_index.xml", "haha")
print("fileNme is {}".format(fileNAME))

file = "/home/UWO/xshuwei/DeepRegS2V/data/LHV/LHV-06/pre06_sweep02/LHV06_pre06_sweep02.xml"
data_tree = ET.parse(file)
root = data_tree.getroot()
cases_metadata = root.find('fixed_image')
print(cases_metadata.find("directory").text)

import platform
platform.system()

path = "/home/UWO/xshuwei/DeepRegS2V\\Registration\\Deepcode\\test\\Pre-Ablation_01.mha"
print(path)
if platform.system() == 'Linux':
    # path.replace("\\\\", '\\')
    path  = '/'.join(path.split('\\'))
    print("haah")
print(path)

project_dir = os.getcwd()
output_dir = os.path.join(project_dir, "src/outputs")
print(output_dir)
project_dir = os.getcwd()
output_dir = os.path.join(project_dir, "src/outputs")
print(output_dir)
now = datetime.now()
now_str = now.strftime('%m%d_%H%M%S')
print('now_str: {}'.format(now_str))
fileName = os.path.join(output_dir, '{}.txt'.format(now_str))
print(fileName)
readFile = open(os.path.join(output_dir, '{}.txt'.format(now_str)), "w")



