import json
import h5py
import pickle
#number of cluster points
cluster_num=1000000
prefix_bow_dir='data/feat_bow/'
shot_frame_file='data/shot_frame_201.json'
shot_frame={}
#bow features of whole videos
bow_feat=[]
#encode the shot_name to number
shot_2_ix={}
ix_2_shot={}
#inverted index
inverted_ix={}
inverted_ix_save='data/inverted_ix.pkl'
shot_2_ix_save='data/shot_2_ix.pkl'
ix_2_shot_save='data/ix_2_shot.pkl'

def init() :
    #load bow features of 244 videos to bow_feat list
    for i in range(244):
        bow_name = 'bow_' + str(i) + '.h5'
        f = h5py.File(prefix_bow_dir + bow_name, 'r')
        bow_value = f['data'].value
        bow_feat.append(bow_value)
        f.close()
    #init the inverted index
    for i in range(cluster_num):
        inverted_ix[i]={}
    inverted_ix[cluster_num]={}
    # encode the shot_name to number
    for i, term in enumerate(shot_frame):
        ix_2_shot[i] = term['name']
        shot_2_ix[term['name']] = i
    pickle.dump(ix_2_shot,open(ix_2_shot_save,'wb'))
    pickle.dump(shot_2_ix,open(shot_2_ix_save,'wb'))

#update inverted index, according to the given shot_id and feat
def inverted_update(shot_id, feat):
    for ix in feat :
        if inverted_ix[ix].has_key(shot_id) :
            inverted_ix[ix][shot_id]=inverted_ix[ix][shot_id]+1
        else :
            inverted_ix[ix][shot_id]=1


def main():
    flag=-1
    for _,term in enumerate(shot_frame):
        name = term['name']
        images_ix = term['images_ix']
        video_index = int(term['video_index'])
        for img_term in images_ix:
            img_term = int(img_term)
            shot_id = shot_2_ix[name]
            feat = bow_feat[video_index][img_term]
            inverted_update(shot_id, feat)
        if flag!=video_index:
            flag=video_index
            print('processing the video : %d' % video_index)
            print('iteration: %d' % _)
    pickle.dump(inverted_ix,open(inverted_ix_save, 'wb'))


shot_frame = json.load(open(shot_frame_file, 'r'))
init()
main()
