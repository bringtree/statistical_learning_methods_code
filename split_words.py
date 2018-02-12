import jieba
print("hello")

# import os
# pwd = "/Users/huangpeisong/Desktop/statistical_learning_methods_code/data_pretreatment/raw_data/raw_data/"
#
# allfiles = os.listdir(pwd)
#
# for file in allfiles:
#   with open(pwd+file) as fp:
#     data = fp.readlines()
#   cut_data = [jieba.lcut(v, cut_all=False) for v in data]
#   with open(file,"w+") as fp:
#     fp.writelines(cut_data)