----------------------------------------------------------------------
-- Create CNN and criterion to optimize.
--
-- Hamid Hassanzadeh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'math'
require 'LSTM'
--require 'Dropout' -- Hinton dropout technique


if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> setting parameters')


local noutputs = 1

local kw = 11
local kh = 4
local kt = opt.order+1
local seq_len = 35
local num_kernels = 5
local lstm_layers={}
for layer_size in string.gmatch(opt.lstm_layers,"%d+") do
	lstm_layers[#lstm_layers+1]=layer_size*1
end
----------------------------------------------------------------------
local model = nn.Sequential()
local model_name ="model.net"

if opt.model == 'CNN' then
	local pool_stride  = math.floor(seq_len-kw+1) 
	local fmap_numel= num_kernels*math.floor((seq_len-kw+1)/pool_stride)
	model:add(nn.SpatialConvolution(1,num_kernels,kw,kh));
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(pool_stride,1));

	model:add(nn.View(fmap_numel))
	model:add(nn.Linear(fmap_numel,32))
	model:add(nn.Dropout(opt.dropout))
	model:add(nn.ReLU())
	model:add(nn.Linear(32,1))

   
	criterion = nn.MSECriterion()
	model_name = "CNN"
elseif opt.model == 'CNN_POOL' then
	local pool_stride  = 2 
	print(sys.COLORS.red ..  '==> construct CNN_POOL')
	local fmap_numel= num_kernels*math.floor((seq_len-kw+1)/pool_stride)
	model:add(nn.SpatialConvolution(1,num_kernels,kw,kh));
	model:add(nn.ReLU())
	model:add(nn.Dropout(opt.dropout))
	model:add(nn.SpatialMaxPooling(pool_stride,1));

	model:add(nn.View(fmap_numel))
	model:add(nn.Linear(fmap_numel,32))
	model:add(nn.Dropout(opt.dropout))
	model:add(nn.Tanh())
	model:add(nn.Linear(32,1))

	criterion = nn.MSECriterion()
	model_name = "CNN_POOL"
elseif opt.model == 'CNN3D' then
	print(sys.COLORS.red ..  '==> construct CNN3D')	
	local pool_stride  = 2 
	local fmap_numel= num_kernels*(opt.order+1 -kt+1)*torch.floor((seq_len-kw+1)/pool_stride)
	model:add(nn.VolumetricConvolution(1, num_kernels,kt, kw,kh))
	model:add(nn.ReLU())
	model:add(nn.VolumetricMaxPooling(1, pool_stride, 1))

	model:add(nn.View(fmap_numel))

    --[[
    model:add(nn.Linear(fmap_numel, 100))
    model:add(nn.Dropout(opt.dropout))
	model:add(nn.Tanh())
	model:add(nn.Linear(100,1))
	]]
    model:add(nn.Linear(fmap_numel, 150))
    model:add(nn.Dropout(opt.dropout))
	model:add(nn.Threshold(0,1e-6))
	model:add(nn.Linear(150,1))

	criterion = nn.MSECriterion()
	model_name = "CNN3D"
elseif opt.model == 'CNN_LSTM' then
	print(sys.COLORS.red ..  '==> construct CNN_LSTM')
	model:add(nn.SpatialConvolution(1,num_kernels,kw,kh));
	model:add(nn.ReLU())

	model:add(nn.View(-1,num_kernels,seq_len-kw+1):setNumInputDims(4))
	model:add(nn.Transpose({2,3}))
	for i=1,#lstm_layers do
		local lstm_inp_dim
		local lstm_outp_dim = lstm_layers[i]
		if (i==1) then 
			lstm_inp_dim = num_kernels
		else
			lstm_inp_dim = lstm_layers[i-1]
		end
		local rnn
		rnn = nn.LSTM(lstm_inp_dim,lstm_outp_dim)
		model:add(rnn)
		model:add(nn.Dropout(opt.dropout))
	end
	model:add(nn.SplitTable(2,3))
	model:add(nn.SelectTable(-1))
	model:add(nn.Linear (lstm_layers[#lstm_layers],1))

	criterion = nn.MSECriterion()
	   
	model_name = "CNN_LSTM"

end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

-- return package:
return {
   model = model,
   criterion = criterion,
   model_name = model_name,
}

