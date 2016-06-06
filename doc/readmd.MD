# ins 任务

## 概述
1. 对224个视频进行图像采样，设定每隔1秒采样一张图像。
2. 用torch，vgg19网络结构，提取所有图像的conv5层的特征，即每幅图像生成196个512维的点，作为局部特征点。
3. 从全体图像中的局部特征点中，平均采样出一小部分，利用akm训练聚类中心。k=1000000.
4. 根据聚类中心，计算每个shot的bow特征。建立倒排表。
5. 对于固定的query，计算bow特征，检索倒排表，求出最匹配的shot。

## 功能

- prepro.MD， 数据预处理


- feat_extrator_threads.MD， 提取conv5特征
- cls_train_data.MD， 生成聚类训练数据
- fastcluster.MD， 聚类
- /evaluation， 评价和计算MAP