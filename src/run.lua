----------------------------------------------------------------------
-- Train a ConvNet on faces.
--
-- original: Clement Farabet
-- new version by: E. Culurciello 
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'pl'
require 'trepl'
require 'torch'	-- torch
require 'image'	-- to visualize the dataset
require 'nn'		-- provides all sorts of trainable modules/layers
require 'os'

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
	-r,--learningRate		 (default 3e-3)		  learning rate
	-d,--learningRateDecay  (default 1e-7)		  learning rate decay (in # samples)
	-w,--weightDecay		  (default 1e-5)		  L2 penalty on the weights
	-m,--momentum			  (default 0.1)			momentum
	-o,--dropout				(default 0.2)			dropout amount
	-b,--batchSize			 (default 40)			batch size
	-t,--threads				(default 2)			  number of threads
	-p,--type					(default float)		 float or cuda
	-i,--devid				  (default 1)			  device ID (if using CUDA)
		--model				  (default CNN)			network model
		--optMethod			 (default sgd)			optimization method
		--data_dir			(default /nv/hp16/hhassanzadeh3/data/Projects/DeepBind/data/)   location of pbm data
		--pbm			(default Cbf1)    name of pbm experiment
		--order		(default 1)   order
		--grad_clip	(default 10) to avoid gradient explosion
		--lstm_layers (default 30,20)  layers' sizes of lstm
		--output_dir (default /nv/hp16/hhassanzadeh3/data/Projects/DeepBind/Results/) Prediction/targets store here
]]

if (opt.order > 1) then
	print ('Error: order >1 is not implemented yet')
	os.exit(1);
end

local arch=''
if (opt.lstm_layers ~= '-') then
	arch='_lstm_'..opt.lstm_layers
end

opt.train_file = opt.data_dir .. opt.pbm .. '/' .. opt.pbm .. '_deBruijn_v1.txt'
opt.test_file  = opt.data_dir .. opt.pbm .. '/' .. opt.pbm .. '_deBruijn_v2.txt'

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(7)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
	print(sys.COLORS.red ..  '==> switching to CUDA')
	require 'cunn'
	cutorch.setDevice(opt.devid)
	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

data  = require 'data'
local train = require 'train'
local test  = require 'test'
------------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

print (opt)
local best_val_cor=-1,selected_test_cor,selected_train_cor
local safe = false;
for i=1,100 do
	train()
	local cor_test,cor_val,cor_train,target_train,preds_train, target_test,preds_test ,class_test=test()
	print('Spearman corr (test/train/val): ' .. torch.round(cor_test*1000)/10.0 .. '%/' .. torch.round(cor_train*1000)/10.0 .. '%/' .. torch.round(cor_val*1000)/10.0 .. '%')

	if (cor_train > .05) then safe = true; end
	--if (i>5 and not safe) then break; end
	if (cor_val>=best_val_cor) then 
		best_val_cor = cor_val
		selected_test_cor = cor_test
		selected_train_cor = cor_train

		file,msg=io.open(opt.output_dir .. '/' .. 'test_'.. opt.pbm ..'_' .. opt.model.. '_lr_'.. opt.learningRate ..'_dr_'.. opt.learningRateDecay .. '_wd_' .. opt.weightDecay ..'_do_' ..opt.dropout .. '_bs_'..opt.batchSize .. arch..'.txt','w')
		if (not file) then print (msg) end
		for j=1,target_test:size(1) do
			local str=target_test[j].."\t" ..preds_test[j].."\t"..class_test[j] .."\n"
			file:write(str)
		end
		io.close(file)

		file=io.open(opt.output_dir .. '/'  .. 'train_'.. opt.pbm ..'_' .. opt.model.. '_lr_'.. opt.learningRate ..'_dr_'.. opt.learningRateDecay .. '_wd_' .. opt.weightDecay ..'_do_' ..opt.dropout .. '_bs_'..opt.batchSize .. arch..'.txt','w')
		for j=1,target_train:size(1) do
			file:write(target_train[j].."\t" ..preds_train[j].. "\n")
		end
		io.close(file)

	end

end


print ('Final spearman corr (test/train/val): ' .. torch.round(selected_test_cor*1000)/10.0 ..'%/' .. torch.round(selected_train_cor*1000)/10.0 .. '%/'.. torch.round(best_val_cor*1000)/10.0 .. '%')

return data
