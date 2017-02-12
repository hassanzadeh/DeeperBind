require 'torch'	-- torch
require 'nnx'		-- provides a normalization operator

print(sys.COLORS.red ..  '==> loading dataset')
-- see if the file exists
local function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
local function read_PBM(file)
	local data={}
	print (sys.COLORS.red .. '==> Reading file: ' .. file);
	if not file_exists(file) then
		print ('File not found: ' .. file )
		os.exit(1) 
	end
	for line in io.lines(file) do 
		local seq,int=line:match("^([^%s]+)%s+([^%s]+)$")

		if (int~= nil) then
			data[#data+ 1] = {seq,int*1}
		end
  	end
	--table.sort (data,function (a,b) return a[2]>b[2] ; end)
	--shuffle the data
	for i=1,#data do
		j=torch.random(#data)
		data[i],data[j]=data[j],data[i]
	end
  	return data
end

local function seq2atcg(seq,seq_len)
	seq=string.upper(seq)
	atcg=torch.Tensor(4,seq_len):fill(0)
	for row,letter  in ipairs({'A','C','G','T'}) do  --Warning the order is important otherwise reversecomp does not work
		local index=0
		while (true) do	
			index=string.find(seq, letter, index)
			if (index ~= nil and index <= seq_len) then
				atcg[{row,index}] = 1
				index = index + 1
			else 
				break
			end
		end
	end
	return atcg
end


local function findClass(data) 
	local ints = torch.Tensor(#data)
	for i=1,#data do
		ints[i]=data[i][2]
	end
	--ints:add(-ints:min()+1):log()
	ints:add(-ints:mean())
	ints:div(ints:std())
	local median,_ = ints:median()
	local MAD,_ =  torch.median(torch.abs(torch.add(ints,-median[1])))
	local sigma = MAD[1]/0.6745
	local thres = (median +4*sigma) 
	local count = ints:gt(thres[1]):sum()

	for i=1,#data do
		if (ints[i]>=thres[1]) then
			data[i][3] = 2
		else
			data[i][3]=1
		end
	end
	return thres,count
end

local train_raw= read_PBM(opt.train_file)
local test_raw= read_PBM(opt.test_file)

local _,train_pos_count = findClass(train_raw)
local _,test_pos_count = findClass(test_raw)

print ('#Pos: Train/Test ' .. train_pos_count ..' ' .. test_pos_count)

local trsize = torch.round(#train_raw*.7) 
local vasize= #train_raw - trsize
local tesize= #test_raw 

local int_tensor = torch.Tensor(#train_raw):fill(0)
local train_int_tensor =torch.Tensor(trsize):fill(0)
local valid_int_tensor =torch.Tensor(vasize):fill(0)
local test_int_tensor = torch.Tensor(tesize):fill(0)

local train_class_tensor =torch.LongTensor(trsize):fill(0)
local valid_class_tensor =torch.LongTensor(vasize):fill(0)
local test_class_tensor = torch.LongTensor(tesize):fill(0)

for i=1,#train_raw do
	int_tensor[i]=train_raw[i][2]	
end
int_tensor:add(-int_tensor:mean()):div(int_tensor:std())
for i=1,tesize do
	test_int_tensor[i]=test_raw[i][2]	
end
test_int_tensor:add(-test_int_tensor:mean()):div(test_int_tensor:std())

local train_input = torch.Tensor()
local valid_input = torch.Tensor()
local test_input = torch.Tensor()

if (opt.model == 'CNN' or opt.model=='CNN_LSTM' or opt.model == 'CNN_POOL') then
	train_input = torch.Tensor(trsize,4,35):fill(0)
	valid_input = torch.Tensor(vasize,4,35):fill(0)
	test_input = torch.Tensor(tesize,4,35):fill(0)
elseif (opt.model == 'CNN3D') then
	train_input = torch.Tensor(trsize,opt.order+1,4,35):fill(0)
	valid_input = torch.Tensor(vasize,opt.order+1,4,35):fill(0)
	test_input= torch.Tensor(tesize,opt.order+1,4,35):fill(0)
end

print ('Train size: ' .. trsize)
print ('Validation size: '.. vasize)
print ('Test size: '.. tesize)

for i=1,trsize do
	train_int_tensor[i]=int_tensor[i]
	train_class_tensor[i]=train_raw[i][3]
	if (opt.model == 'CNN' or opt.model=='CNN_LSTM' or opt.model == 'CNN_POOL') then
		train_input[i] = seq2atcg(train_raw[i][1],35)
	elseif (opt.model == 'CNN3D') then
		--Warning should be updated for dynamic i
		train_input[i][1] = seq2atcg(train_raw[i][1],35)
		train_input[{{i},{2},{},{1,34}}] = train_input[{{i},{1},{},{2,35}}]
	end
end
for i=trsize+1,#train_raw do
	valid_int_tensor[i-trsize]=int_tensor[i]
	valid_class_tensor[i-trsize]=train_raw[i][3]
	if (opt.model == 'CNN' or opt.model=='CNN_LSTM' or opt.model == 'CNN_POOL') then
		valid_input[i-trsize]=seq2atcg(train_raw[i][1],35)
	elseif (opt.model == 'CNN3D') then
		--Warning should be updated for dynamic i
		valid_input[i-trsize][1]=seq2atcg(train_raw[i][1],35)
		valid_input[{{i-trsize},{2},{},{1,34}}]=train_input[{{i},{1},{},{2,35}}]
	end
end

for i=1,tesize do
	test_class_tensor[i]=test_raw[i][3]
	if (opt.model == 'CNN' or opt.model=='CNN_LSTM' or opt.model == 'CNN_POOL') then
		test_input[i]=seq2atcg(test_raw[i][1],35)
	elseif (opt.model == 'CNN3D') then
		test_input[i][1]=seq2atcg(test_raw[i][1],35)
		test_input[{{i},{2},{},{1,34}}]=test_input[{{i},{1},{},{2,35}}]
	end
end

-- create train set:
local trainData = {
	seq = train_input,
	int = train_int_tensor,
	class=train_class_tensor,
	size = function() return int:size(1) end
}

-- create validation set:
local validData = {
	seq = valid_input,
	int = valid_int_tensor,
	class=valid_class_tensor,
	size = function() return int:size(1) end
}
--create test set:
local testData = {
	seq = test_input,
	int = test_int_tensor,
	class=test_class_tensor,
	size = function() return int:size(1) end
}

-- remove from memory temp image files:
int_tensor= nil

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> preprocessing data')
-- Exports
return {
	train= trainData,
	valid=validData,
	test= testData
}
