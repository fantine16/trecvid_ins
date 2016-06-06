require 'hdf5'
featTotal=nil
stride=70
load_prefix=''

function getFeatByFileName(fileName)
    local file=hdf5.open(fileName,'r')
    local data = file:read('data'):all()
    local dim=data:size()
    local index=torch.range(1,dim[1]/stride):long()
    index = index * stride
    local feat = data:index(1,index):contiguous()
    file:close()
    data=nil
    collectgarbage()
    return feat
end

--feat=getFeatByFileName('1.h5')

for i=1,223 do
    local fileName = load_prefix .. 'video_' .. vidtostring(i) .. '.h5'
    local feat = getFeatByFileName(fileName)
    if featTotal == nil then
        featTotal = feat
    else
        featTotal=torch.cat(featTotal,feat,1)
    end
end

torch.type(featTotal)

out=hdf5.open('rawData.h5','w')
out:write('data',featTotal)
out:close()
