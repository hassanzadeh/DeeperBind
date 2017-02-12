require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'image'
require 'corr'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local criterion = t.criterion

local debug = model:get(6)

-- Batch test:

local x_train,x_test,x_valid
local targets_train ,targets_test ,targets_valid

targets_train =data.train.int:clone()
targets_test =data.test.int:clone()
targets_valid =data.valid.int:clone()

class_train =data.train.class:clone()
class_test =data.test.class:clone()
class_valid =data.valid.class:clone()

if opt.model == 'CNN' or opt.model == 'CNN_LSTM' or opt.model == 'CNN_POOL' then
	x_train= torch.Tensor(data.train.seq:size(1) , 1,data.train.seq:size(2) , data.train.seq:size(3)) 
	x_train[{{},1,{},{}}]=data.train.seq

	x_test= torch.Tensor(data.test.seq:size(1) , 1,data.test.seq:size(2) , data.test.seq:size(3)) 
	x_test[{{},1,{},{}}]=data.test.seq

	x_val= torch.Tensor(data.valid.seq:size(1) , 1,data.valid.seq:size(2) , data.valid.seq:size(3)) 
	x_val[{{},1,{},{}}]=data.valid.seq
elseif opt.model == 'CNN3D' then
	x_train= torch.Tensor(data.train.seq:size(1) , 1,data.train.seq:size(2)  , data.train.seq:size(3), data.train.seq:size(4)) 
	x_train[{{},1,{},{},{}}]=data.train.seq

	x_test= torch.Tensor(data.test.seq:size(1) , 1,data.test.seq:size(2)  , data.test.seq:size(3), data.test.seq:size(4)) 
	x_test[{{},1,{},{},{}}]=data.test.seq

	x_valid= torch.Tensor(data.valid.seq:size(1) , 1,data.valid.seq:size(2)  , data.valid.seq:size(3), data.valid.seq:size(4)) 
	x_valid[{{},1,{},{},{}}]=data.valid.seq
end

if opt.type == 'cuda' then 
	x_train=x_train:cuda()
	targets_train = targets_train:cuda()

	x_valid=x_valid:cuda()
	targets_valid= targets_valid:cuda()

	x_test=x_test:cuda()
	targets_test= targets_test:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test()
	model:evluate()
	local time = sys.clock()

		-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	local preds_test = model:forward(x_test):clone():resize(x_test:size(1))
	local preds_val= model:forward(x_val):clone():resize(x_val:size(1))
	local preds_train = model:forward(x_train):clone():resize(x_train:size(1))
	
	local cor_train,cor_val,cor_test 
	local err_test ,err_val 
	cor_train = spearman(preds_train,targets_train)
	cor_val= spearman(preds_val,targets_valid)
	cor_test = spearman(preds_test,targets_test)
	err_test =  torch.pow(torch.norm(preds_test-targets_test),2)
	err_val =  torch.pow(torch.norm(preds_val-targets_valid),2)
	print ("Test/Val error: " .. torch.trunc(err_test) .. "/" .. torch.trunc(err_val ))

	return cor_test,cor_val,cor_train,targets_train,preds_train,targets_test,preds_test,class_test

end

-- Export:
return test

