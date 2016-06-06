rm *.h5
rm *.txt

echo 'transform images format to jpg'
python format2jpg.py
echo 'extract vgg19 conv5 features'
th feat_extrator.lua
echo 'calculate bow features'
python bow_extrator.py
echo 'search'
python search.py
echo 'eval'
./trec_eval_video/trec_eval -q -c  ins.search.qrels.tv15 predict.txt 1000 | grep 'map'
