import numpy as np
import h5py
import fastann
bow_save_prefix="data/feat_bow/"
video_feat_file=open('video_feat_path.txt','r')
paths=[]
clst_file=h5py.File('clst.h5','r')
print('loading clusters...')
clst_point=clst_file['clusters'].value
print('complete')
print('buiding kd tree...')
nno_kdt = fastann.build_kdtree(clst_point, 8, 768)
print('complete')
while True:
    line = video_feat_file.readline().strip()
    if line:
        paths.append(line)
    else:
        break

for path in paths:
    print(path)
    file_name=path.split('/')[len(path.split('/'))-1]
    bow_name='bow'+file_name[5:]
    video_file=h5py.File(path,'r')
    bow_file=h5py.File(bow_save_prefix + bow_name,'w')
    print('loading video h5 file:'+file_name)
    data=video_file['data'].value
    #print('dimention of data:'+data.shape)
    num_image=data.shape[0]
    num_point=data.shape[1]
    dim=data.shape[2]
    data.resize(num_image*num_point,dim)
    print('compute bow features')
    argmins_kdt, mins_kdt = nno_kdt.search_nn(data)
    argmins_kdt.resize(num_image,num_point)
    print('saving bow features to:'+bow_save_prefix + bow_name)
    bow_file.create_dataset('data',data=argmins_kdt)
    bow_file.flush()
    bow_file.close()
    video_file.close()


print('complete')
