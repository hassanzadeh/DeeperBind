
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'LSTM'
require 'optim'

local opt={}

local model = nn.Sequential()
local model_name ="parity"
local D,H=1,100
local dr = 0.
local batchSize=200
local lr_decay_every= 5
local optim_config = {learningRate = .002}
local classes={'1','2','3'}
local dims={D,H}
local lr_decay_factor = .5
local gpu =true --false 
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
model:add(nn.Linear (dims[#dims],1))

criterion = nn.MSECriterion()

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
	criterion:cuda()
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

	local y= model:forward(x)
	local E = criterion:forward(y,yt)
	-- estimate df/dW
	local dE_dy = criterion:backward(y,yt)   
	model:backward(x,dE_dy)

	grad_params:clamp(-grad_clip, grad_clip)

	return E, grad_params
end


test_input= torch.Tensor (1000,25,1):random(10):add(-1)
test_target=  torch.Tensor(1000):copy(test_input:sum(2):reshape(1000,1))
if gpu then 
   test_input = test_input:cuda()
   test_target = test_target:cuda()
end
for epoch = 1 , 1000 do
	local total = 0
	local sum_err = 0
	print ('=============================================')
	for t = 1,200 do
		-- create mini batch
		local idx = 1
		local T=torch.random(30)
		if (T==25) then T=torch.random(30) end
		x:resize(batchSize,T,1)
		yt:resize(batchSize)
		for i = t,t+batchSize-1 do
			x[idx] = torch.Tensor(T,1):random(10):add(-1)
			yt[idx] = x[idx]:sum(1)
			idx = idx + 1
		end
		--local _, err= optim.adam(f, params, optim_config)
		local _ , err= optim.sgd(f, params, optim_config)
		sum_err = sum_err + err[1]/T
		total = total + 1
	end
	print ('Train error epoch (' .. epoch .. '): ' .. sum_err /total)
	if epoch % lr_decay_every == 0 then
		local old_lr = optim_config.learningRate
		--optim_config = {learningRate = old_lr * lr_decay_factor}
		optim_config = {learningRate = old_lr  }
	end

	local y= model:forward(test_input)
	local E = criterion:forward(y,test_target)
	print ('Test error ' .. E/test_input:size(2))
	print (torch.cat(y[{{20,30}}],test_target[{{20,30}}] ,2))
end
	










-- return package:
return {
   model = model,
   loss = loss,
   model_name = model_name,
}

