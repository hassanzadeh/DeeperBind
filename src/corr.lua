require 'os'

function spearman(y,yt)
	
	local list ={}
	if (y:size(1) ~= yt:size(1)) then os.exit(1) end

	for i=1,y:size(1) do
		list[#list+1]={y[i],yt[i]}
	end

		

	local function aux_func(t) -- auxiliary function
		return (t == 1 and 0) or (t*t - 1) * t / 12
	end

	for _, v in pairs(list) do v.r = {} end
	local T, S = {}, {}
	-- compute the rank
	for k = 1, 2 do
 		table.sort(list, function(u,v) return u[k]<v[k] end)
		local same = 1
		T[k] = 0
		for i = 2, #list + 1 do
			if i <= #list and list[i-1][k] == list[i][k] then 
				same = same + 1
			else
				local rank = (i-1) * 2 - same + 1
				for j = i - same, i - 1 do list[j].r[k] = rank end
				if same > 1 then T[k], same = T[k] + aux_func(same), 1 end
			end
		end
		S[k] = aux_func(#list) - T[k]
	end
		-- compute the coefficient
	local sum = 0
	for _, v in pairs(list) do -- TODO: use nested loops to reduce loss of precision
		local t = (v.r[1] - v.r[2]) / 2
		sum = sum + t * t
	end
	return (S[1] + S[2] - sum) / 2 / math.sqrt(S[1] * S[2])
end
