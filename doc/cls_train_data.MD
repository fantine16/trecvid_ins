# cls_train_data

## 准备工作
已经提取了每个视频的采样图像的conv5的特征，每个图像有196个512维的点。

## 功能 
现在要对这些点进行采样，提取一小部分，为AKM聚类作准备。


## 运行参数

- stride ,每stride个点，提取1个点。
- load_prefix, 视频的conv5特征文件的所在目录。

## 运行结果

产生rawData.h5文件，目录/data下，存的是2D tensor, 大小，(N, 512)
N表示采样点的总数量。生成的rawData.h5文件可以直接让akm读取，聚类。