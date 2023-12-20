import os
cwd = os.getcwd()
print("current directory is: ", cwd)

fileNAME = os.path.join(cwd, "dataset_index.xml", "haha")
print("fileNme is {}".format(fileNAME))