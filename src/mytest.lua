require 'torch'
require 'nn'
require 'image'

local function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end


local function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
  	seq,int=line:match("^([^%s]+)%s+([^%s]+)$")
	if (int~= nil) then
		lines[#lines + 1] = line
	end
  end
  print (#lines)
  return lines
end
local function seq2atcg(seq)

	seq=string.upper(seq)
	atcg=torch.Tensor(4,seq:len()):fill(0)
	for row,letter  in pairs({'A','T','C','G'}) do
		index=0
		repeat 
			index=string.find(seq, letter, index)
			if (index ~= nil) then
				atcg[{row,index}] = 1
				index = index + 1
			else 
				break
			end
		until (index > 35 or index == nil)
	end
	return atcg
end

local lines = lines_from('/nv/hp16/hhassanzadeh3/data/Projects/DeepBind/data/Cbf/Cbf1_deBruijn_v1.txt')
seq='gtcGGAT'
atcg = seq2atcg(seq)
print (atcg)
print (seq)


