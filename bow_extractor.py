import os
import h5py
import fastann
bow_save_dir="data/feat_bow1/"
video_feat_dir="/datacenter/2/tv16ins_feat_video/"
paths = os.listdir(video_feat_dir)
clst_file=h5py.File('data/clst.h5','r')
print('loading clusters...')
clst_point=clst_file['clusters'].value
print('complete')
print('buiding kd tree...')
nno_kdt = fastann.build_kdtree(clst_point, 8, 768)
print('complete')


for path in paths:
    print(path)
    bow_name='bow'+path[5:]
    video_file=h5py.File(video_feat_dir+path,'r')
    bow_file=h5py.File(bow_save_dir + bow_name,'w')
    print('loading video h5 file:'+path)
    data=video_file['data'].value
    #print('dimention of data:'+data.shape)
    num_image=data.shape[0]
    num_point=data.shape[1]
    dim=data.shape[2]
    data.resize(num_image*num_point,dim)
    print('compute bow features')
    argmins_kdt, mins_kdt = nno_kdt.search_nn(data)
    argmins_kdt.resize(num_image,num_point)
    print('saving bow features to:'+bow_save_dir + bow_name)
    bow_file.create_dataset('data',data=argmins_kdt)
    bow_file.flush()
    bow_file.close()
    video_file.close()


print('complete')
