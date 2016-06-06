require 'hdf5'
featTotal=nil
num=70
stride=70
flag=1


function getFeatByFileName(fileName)
    local file=hdf5.open(fileName,'r')
    local data = file:read('data'):all()
    local dim=data:size()
    local index=torch.range(1,dim[1]/70):long()
    index = index * 70
    local feat = data:index(1,index):contiguous()
    file:close()
    data=nil
    collectgarbage()
    return feat
end

--feat=getFeatByFileName('1.h5')

for i=1,2 do
    local fileName = tostring(i) .. '.h5'
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
