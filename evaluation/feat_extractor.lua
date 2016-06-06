require 'torch'
require 'nn'
require 'image'
require 'cudnn'
require 'cunn'
require 'loadcaffe'
require 'hdf5'
require 'cutorch'
cjson = require 'cjson'
backend='cudnn'
model_weights='../model_weights/VGG_ILSVRC_19_layers.caffemodel'
model_proto='../model_weights/VGG_ILSVRC_19_layers_deploy.prototxt'
image_path='query_image.txt'
save_prefix=''
batch_size=20
loadSize={3,256,256}
test_model=false
GPU_ID=1


function loadImage(path)
    local input
    if test_model then --不读取图片，生成图像，用于测试
        input=torch.FloatTensor(3,256,256):zero()
        return input
    end
   input = image.load(path, 3, 'float')
   input = image.scale(input,256,256)
   
    collectgarbage()
   return input
end

function prepro(imgs)
    local h,w = imgs:size(3), imgs:size(4)
    local cnn_input_size = 224
    local xoff, yoff
    xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]    
    local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1)
    imgs:add(-1, vgg_mean:expandAs(imgs))
    imgs = imgs:cuda()
    return imgs
end

function getPretrainedModel()
    local model = loadcaffe.load(model_proto,model_weights,backend)
    model:evaluate()
    model = model:cuda()
    collectgarbage()
    return model
end

function tensor_tran(x)
    local y=x:transpose(2,4):clone():resize(x:size(1),x:size(3)*x:size(4),x:size(2))
    return y
end

function getbatch(img_paths)
    local img_num=#img_paths
    local out=torch.FloatTensor(img_num,loadSize[1],loadSize[2],loadSize[3])
    for i,path in pairs(img_paths) do 
        local img=loadImage(path)
       -- print(img:size())
        out[{{i,i}}]=img
        if i%1==0 then
            print("当前batch有" .. img_num .. "张图片，正在读取第" .. i .. "张")
        end
    end
    out=prepro(out)
    collectgarbage()
    return out
end

function extractByVideo(video_index,img_paths,model,query)
    local img_num=#img_paths
    local batch_iter=math.floor(img_num/batch_size) --总batch的数量
    local res_imgs=img_num%batch_size --剩余的未处理的图像数量
    local i ,j
    local count=1
    local batch_img,outputs,feat,batch_img_paths
    local feat_all=torch.FloatTensor(img_num,14*14,512):zero()
    local output_h5
    if query then
        output_h5=save_prefix .. "video_query" .. ".h5"
    else
        output_h5=save_prefix .. "video_" .. video_index ..".h5"
    end
    print("第" .. video_index .. "个视频，采样图像数量：" .. img_num)
    
    for i=1,batch_iter do 
        batch_img_paths={}
        for j=1,batch_size do 
            table.insert(batch_img_paths,img_paths[count])
            count=count+1
        end
        print("第" .. video_index .. "个视频，共" .. batch_iter .. "个batch，当前batch序号：" .. i )
        batch_img=getbatch(batch_img_paths)
        model:forward(batch_img)        
        outputs=model.modules[35].output
        feat=tensor_tran(outputs)
        feat_all[{{(i-1)*batch_size+1,i*batch_size}}]=feat:float() --29行
    end
   if res_imgs>=1 then
        batch_img_paths={}
        for i=1,res_imgs do
            table.insert(batch_img_paths,img_paths[count])
            count=count+1
        end
        batch_img=getbatch(batch_img_paths)
        model:forward(batch_img)
        outputs=model.modules[35].output
        feat=tensor_tran(outputs)
        feat_all[{{batch_iter*batch_size+1,img_num}}]=feat:float()
    end
    local myfile=hdf5.open(output_h5,'w')
    myfile:write('data',feat_all)
    myfile:close()
    collectgarbage()
end

function main()
	file=io.open('query_images.txt','r')
	local img_paths={}
	for l in file:lines() do
        if string.len(l)>1 then  
            table.insert(img_paths,l)
        end		
	end
	file.close()
	model=getPretrainedModel()
	extractByVideo(1,img_paths,model,  true)
end

main()