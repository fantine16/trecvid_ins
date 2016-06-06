# feat_extrator_threads

提取frame的vgg19的conv5的特征，并且按照video，保存成hdf5格式。

## 运行参数

- batch_size, 提取特征的batch的大小
- nThreads，多线程读取图像的线程数量
- GPU_ID, 指定训练gpu的设备
- video_start, 处理video的起始序号
- video_end，处理video的终止序号
- save_prefix, 保存的h5文件的路径

## loadImage(path)

根据path，从硬盘读取图像，并且resize成 256X256 大小。
- 返回 tensor，大小(3, 256, 256)

## prepro(imgs)

对图像数据进行剪裁和归一化的预处理。
- imgs: 4D tensor (batch_size, 3, 256, 256)，一个batch的图像的数据。
- 返回 tensor，大小(batch_size, 3, 224, 224)，预处理之后的一个batch的图像数据。

## readJson(path)

根据path，读取json文件并且返回。

## splitVideo(dataJson)

根绝dataJson，提取出每个视频的图像的路径列表。

- dataJson， 列表，每个元素描述1个shot的详细的信息。
- 返回 split，table类型
 - key：video的序号
 - value：该video包含的所有的图像的路径的table

## getPretrainedModel()

返回 caffe model

## tensor_tran(x)

对于size是(1000，512，14，14)的4dtensor,转化成(1000,14*14,512)的2dtensor, 并返回。

## getbatch(img_paths)

img_path是图像的路径列表，多线程，读取图像。

- 返回batch的tensor,(batch_size, 3, 224, 224)

## extractByVideo(video_index,img_paths,model)

### 输入参数

- video_index，video的序号
- img_paths，video包含的所有的图像的路径列表
- model， caffe model

### 功能

把video的所有图片的conv5的特征，存到hdf5文件中。img_num是该video包含的图像的数量。

hdf5文件，/data目录下，存的是3D tensor，(img_num, 196, 512)

# main()

主函数，循环调用 extractByVideo()

