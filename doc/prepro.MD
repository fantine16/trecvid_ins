# prepro.py

处理 eastenders.masterShotReferenceTable 文件，生成描述shot的json格式的数据。

## 输入参数

- --local, true或者false，表示在本地或者远程的模式。

## 生成json文件

格式，list，471526个元素，表示共有471526个shot。每个元素是dict格式，描述1个shot的具体信息。
每个元素的格式：

- name, shot的唯一标识名称。
- video_index, 所属video的序号
- shot_index，所属shot的序号
- image_ix，包含的图片的序号，list
- image_num，包含的图片数量
- img_path，包含的图片的路径，list
- start_second，开始时间，秒
- end_second，结束时间，秒
