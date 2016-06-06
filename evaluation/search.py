import pickle
import h5py
import numpy as np

inverted_ix_save='../data/inverted_ix.pkl'
ix_2_shot_save='../data/ix_2_shot.pkl'
print('loading ix_2_shot')
ix_2_shot=pickle.load(open(ix_2_shot_save,'r'))
print('loading inverted_ix')
inverted_ix=pickle.load(open(inverted_ix_save,'r'))
print('loading finished')


def search(bow):
    feat={}
    res={}
    for term in bow:
        feat[term]=feat.get(term,0)+1
    for ix in feat:
        bin=feat[ix]
        if len(inverted_ix[ix])>0:
            t=sorted(inverted_ix[ix].iteritems(),key=lambda x:abs(x[1]-bin),reverse=False)
            dist2=t[0][1]
            for term in t:
                if term[1]==dist2:
                    res[term[0]] = res.get(term[0], 0) + 1
                else:
                    break;

    final=sorted(res.iteritems(),key=lambda x:x[1],reverse=True)
    return final




def main():
    f = h5py.File('bow_query.h5', 'r')
    f_path=open('query_images.txt','r')
    f_res=open('predict.txt','w')
    topic={}
    line_num=0
    while True:
        line = f_path.readline().strip()
        if line:
            id=line.split('/')[-1].split('.')[0]
            if topic.has_key(id):
                topic[id].append(line_num)
            else:
                topic[id]=[]
                topic[id].append(line_num)
        else:
            break
        line_num=line_num+1

    data=f['data'].value

    for term in topic:
        bow_query = np.array([])
        for ix in topic[term]:
            bow_query=np.concatenate((data[ix],bow_query))
            #bow_query = data[ix] + bow_query
        shot_id = search(bow_query)
        for j, t in enumerate(shot_id):
            if j < 1000:
                str1 = str(term) + '\tQ0\t' + ix_2_shot[t[0]] + '\t' + str(j + 1) + '\t' + str(t[1]) + '\tSTANDART'
                f_res.write(str1 + '\n')

    f_res.close()
    f.close()
    f_path.close()


main()