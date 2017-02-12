
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'LSTM'
require 'optim'

local opt={}

local model = nn.Sequential()
local model_name ="parity"
local D,H=1,10
local dr = 0.
local batchSize=200
local lr_decay_every= 5
local optim_config = {learningRate = .4}
local classes={'1','2','3'}
local dims={D,H}
local lr_decay_factor = .5
local gpu =false 
grad_clip = 500000

torch.manualSeed(1)

for i=1,#dims-1 do
	local rnn
	rnn = nn.LSTM(dims[i],dims[i+1] )
	--table.insert(self.rnns, rnn)
	model:add(rnn)
	if dr > 0 then
		model:add(nn.Dropout(dr))
	end
end
model:add(nn.SplitTable(1,2))
model:add(nn.SelectTable(-1))
model:add(nn.Linear (dims[#dims],#classes))
model:add(nn.LogSoftMax())

--local loss= nn.CrossEntropyCriterion()
loss = nn.ClassNLLCriterion()

-- This matrix records the current confusion_train across classes
local confusion_train = optim.ConfusionMatrix(classes)
local confusion_test= optim.ConfusionMatrix(classes)

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if gpu then
	print(sys.COLORS.red ..  '==> switching to CUDA')
	require 'cunn'
	local devid=1
	cutorch.setDevice(devid)
	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
	model:cuda()
	loss:cuda()
end


local params, grad_params = model:getParameters()

--Training


x=torch.Tensor()
yt=torch.Tensor()

if gpu then 
   x = x:cuda()
   yt = yt:cuda()
end

-- Loss function that we pass to an optim method
local function f(w)
	assert(w == params)
	grad_params:zero()

	local scores = model:forward(x)
	local E = loss:forward(scores,yt)
	-- estimate df/dW
	local dE_dy = loss:backward(scores,yt)   
	model:backward(x,dE_dy)

	for i = 1,batchSize do
		confusion_train:add(scores[i],yt[i])
	end

	--grad_params:clamp(-grad_clip, grad_clip)

	return E, grad_params
end


test_input= torch.Tensor (1000,100,1):random(2):add(-1)
test_target=  torch.Tensor(1000):copy(test_input:sum(2):reshape(1000,1)):mod(#classes):add(1)

for epoch = 1 , 1000 do
	for t = 1,200 do

		-- create mini batch
		local idx = 1
		local T=torch.random(10)
		x:resize(batchSize,T,1)
		yt:resize(batchSize)
		for i = t,t+batchSize-1 do
			x[idx] = torch.Tensor(T,1):random(2):add(-1)
			yt[idx] = x[idx]:sum(1):mod(#classes):add(1)
			idx = idx + 1
		end
		--local _, err= optim.adam(f, params, optim_config)
		local _, err= optim.sgd(f, params, optim_config)
	end
	if epoch % lr_decay_every == 0 then
		local old_lr = optim_config.learningRate
		--optim_config = {learningRate = old_lr * lr_decay_factor}
		optim_config = {learningRate = old_lr  }
	end


	print ('=============================================')
	print(confusion_train)
	local scores=model:forward(test_input)
	for i=1,test_target:size(1) do
		confusion_test:add(scores[i],test_target[i])
	end
	print(confusion_test)

	confusion_train:zero()
	confusion_test:zero()
end
	










-- return package:
return {
   model = model,
   loss = loss,
   model_name = model_name,
}

