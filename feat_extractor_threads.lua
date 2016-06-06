
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
Threads = require 'threads'
--Threads.serialization('threads.sharedserialize')
model_weights='model_weights/VGG_ILSVRC_19_layers.caffemodel'
model_proto='model_weights/VGG_ILSVRC_19_layers_deploy.prototxt'
input_shot='data/shot_frame_201.json'
save_prefix='data/'
batch_size=100
local loadSize={3,256,256}
test_model=false
video_start=0
video_end=243
GPU_ID=4
nTheads=8 

local function loadImage(path)
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

function readJson(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  collectgarbage()
  return info
end

function splitVideo(dataJson)
    --print(dataJson[1])
    local split={}
    for i,term in pairs(dataJson) do
        if not split[term.video_index] then 
            split[term.video_index]={}
        end
        if #term['img_path']>0 then 
            for _,path in pairs(term['img_path']) do
                table.insert(split[term.video_index],path)
            end
        end         
    end
    collectgarbage()
    return split
end

function getPretrainedModel()
    local model = loadcaffe.load(model_proto,model_weights,backend)
    model:evaluate()
    model = model:cuda()
    collectgarbage()
    return model
end
      
function getbatch(img_paths)
    local img_num=#img_paths
    local img_per=math.floor(img_num/nTheads)
    local img_res=img_num-(nTheads-1)*img_per
    print("当前batch图片数量 : " .. img_num)
    local workers=Threads(nTheads,
        function ()
            require 'image'
            --print(__threadid .. " start thread")
        end,
        function ()
            th_img_paths=img_paths
            th_img_num=#th_img_paths
            th_img_per=img_per
            th_loadImage=loadImage
        end
    )
    local out=torch.FloatTensor(img_num,loadSize[1],loadSize[2],loadSize[3])
    print("分配 " .. nTheads .. " 个线程读取图像" )
    workers:specific(true)
    for i=1,nTheads-1 do 
        workers:addjob(
            i,
           -- getbatch_thread(img_paths,1+(i-1)*img_per,i*img_per),
            function()
                local start_ix = 1+(__threadid-1)*th_img_per
                local end_ix=__threadid*th_img_per
                local img_num=end_ix-start_ix+1
                local out=torch.FloatTensor(img_num,loadSize[1],loadSize[2],loadSize[3])
                print("线程 " .. __threadid .. " 要读取 " .. img_num .. " 张图片")
                for j=start_ix,end_ix do
                    --print(th_img_paths[j])
                    local img=th_loadImage(th_img_paths[j])
                    out[{{j-start_ix+1,j-start_ix+1}}]=img
                end
                return out, start_ix, end_ix 
            end,

            function(img_tensor,start_ix,end_ix)
                out[{{start_ix,end_ix}}]=img_tensor
            end
        )
    end
    if img_res>=1 then
        workers:addjob(
            nTheads, 
            --getbatch_thread(img_paths,(nTheads-1)*img_per+1,img_num),
            function()
                local start_ix = 1+(__threadid-1)*th_img_per
                local end_ix=th_img_num
                local img_num_t=end_ix-start_ix+1
                local out=torch.FloatTensor(img_num_t,loadSize[1],loadSize[2],loadSize[3])
                print("线程 " .. __threadid .. " 要读取 " .. img_num_t .. " 张图片")
                for j=start_ix,end_ix do
                    local img=th_loadImage(th_img_paths[j])
                    out[{{j-start_ix+1,j-start_ix+1}}]=img
                end
                return out, start_ix, end_ix 
            end,
            function(img_tensor,start_ix,end_ix)
                out[{{start_ix,end_ix}}]=img_tensor
            end
        )
    end
    workers:synchronize()
    out=prepro(out)
    collectgarbage()
    return out
end

--对于size是(1000，512，14，14)的4dtensor,转化成(1000,14*14,512)的2dtensor
function tensor_tran(x)
    local y=x:transpose(2,4):clone():resize(x:size(1),x:size(3)*x:size(4),x:size(2))
    return y
end

function extractByVideo(video_index,img_paths,model)
    local img_num=#img_paths
    local batch_iter=math.floor(img_num/batch_size) --总batch的数量
    local res_imgs=img_num%batch_size --剩余的未处理的图像数量
    local i ,j
    local count=1
    local batch_img,outputs,feat,batch_img_paths
    local feat_all=torch.FloatTensor(img_num,14*14,512):zero()
    local output_json=save_prefix .. "video_" .. video_index ..".h5"
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
    local myfile=hdf5.open(output_json,'w')
    myfile:write('data',feat_all)
    myfile:close()
    collectgarbage()
end

function main()
    cutorch.setDevice(GPU_ID)
    dataJson=readJson(input_shot)
    split=splitVideo(dataJson)
    model=getPretrainedModel()
    local i,video_index,img_paths
    for i=video_start,video_end do 
        print("正在处理第 " .. i .. " 个视频")
        video_index= "".. i
        img_paths=split[video_index]
        extractByVideo(video_index,img_paths,model)
        collectgarbage()
    end
end

main()


