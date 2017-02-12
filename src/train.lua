require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'image'

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
--local fwmodel = t.model
local criterion = t.criterion
local model_name = t.model_name


local debug = model:get(7)
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining some tools')

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	weightDecay = opt.weightDecay,
	learningRateDecay = opt.learningRateDecay
}

local function deepCopy(object)
	local lookup_table = {}
	local function _copy(object)
		if type(object) ~= "table" then
			return object
		elseif lookup_table[object] then
			return lookup_table[object]
		end
		local new_table = {}
		lookup_table[object] = new_table
		for index, value in pairs(object) do
			new_table[_copy(index)] = _copy(value)
		end
		return setmetatable(new_table, getmetatable(object))
	end
	return _copy(object)
end
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local x
if opt.model == 'CNN' or opt.model=='CNN_LSTM' or opt.model == 'CNN_POOL' then
	x = torch.Tensor(opt.batchSize,1,data.train.seq:size(2), data.train.seq:size(3))
elseif opt.model == 'CNN3D' then
	x = torch.Tensor(opt.batchSize,1,data.train.seq:size(2), data.train.seq:size(3), data.train.seq:size(4))
end
local yt = torch.Tensor(opt.batchSize)
if opt.type == 'cuda' then 
	x = x:cuda()
	yt = yt:cuda()
end

function rearrange(seq_ten,ints)

	if opt.model == 'CNN3D' then
		print ('Error, this function is not defined for order > 0')
		os.exit(1)
	end
	if (seq_ten:dim() ~= 3 or seq_ten:size(2)~=4) then os.exit(1) end
	local x = torch.Tensor(1,1,seq_ten:size(2), seq_ten:size(3))
	local x_rev = torch.Tensor(1,1,data.train.seq:size(2), data.train.seq:size(3))
	local count = 0

	for i=1,seq_ten:size(1) do
		x[{1,1,{},{}}]=seq_ten[i]
		x_rev = image.flip(image.flip(x,3),4)
		local p = model:forward(x)
		local yt=torch.Tensor({ints[i]})
		err=criterion:forward(p,yt)
		p = model:forward(x_rev)
		err_rev=criterion:forward(p,yt)
		if (err_rev< err) then
			seq_ten[i]=x_rev[{1,1,{},{}}]
			count = count + 1
		end
	end
	print ('Total reversed: ' .. count)
end
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch
local function train()
	model:training()

   -- epoch tracker
	epoch = epoch or 1

   -- local vars
	local time = sys.clock()
	local run_passed = 0
	local mean_dfdx = torch.Tensor():typeAs(w):resizeAs(w):zero()

   -- shuffle at each epoch
	local shuffle = torch.randperm(data.train.int:size(1))
	local err = 0

	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,data.train.int:size(1),opt.batchSize do
      -- disp progress
		--xlua.progress(t, data.train.int:size(1))
		collectgarbage()

      -- batch fits?
		if (t + opt.batchSize - 1) > data.train.int:size(1) then
			break
		end

      -- create mini batch
		local idx = 1
		for i = t,t+opt.batchSize-1 do
			x[idx][1] = data.train.seq[shuffle[i]]
			yt[idx] = data.train.int[shuffle[i]]
			idx = idx + 1
		end

         -- create closure to evaluate f(X) and df/dX
		local eval_E = function(w)
            -- reset gradients
			dE_dw:zero()
	
            -- evaluate function for complete mini batch
			local y = model:forward(x)
			local E = criterion:forward(y,yt)
	
            -- estimate df/dW
			local dE_dy = criterion:backward(y,yt)   
			model:backward(x,dE_dy)

    	        -- update confusion
			---for i = 1,opt.batchSize do !!!!!!!!!! not sure why it was in the loop, commented out for now
			err = err + torch.pow(torch.norm(y-yt),2)
			--end

			--dE_dw:clamp(-opt.grad_clip, opt.grad_clip)
            -- return f and df/dX
			return E,dE_dw
		end

         -- optimize on current mini-batch
		if opt.optMethod == 'sgd' then
			optim.sgd(eval_E, w, optimState)
		elseif opt.optMethod == 'asgd' then
			run_passed = run_passed + 1
			mean_dfdx  = asgd(eval_E, w, run_passed, mean_dfdx, optimState)
		elseif opt.optMethod == 'rmsprop' then
			optim.rmsprop(eval_E, w, optimState)
		end                                


	end
print ("Training error: " .. torch.trunc(err))
   -- time taken
time = sys.clock() - time
time = time / data.train.int:size(1)
--print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- save/log current net
   --local filename = paths.concat(opt.save, model_name)
   --os.execute('mkdir -p ' .. sys.dirname(filename))
   --print('==> saving model to '..filename)
   --model1 = model:clone()
   --netLighter(model1)
   --torch.save(filename, model1)

   -- next epoch
	epoch = epoch + 1
end

-- Export:
return train

