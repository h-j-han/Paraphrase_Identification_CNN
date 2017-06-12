local Conv = torch.class('similarityMeasure.Conv')

function Conv:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.01
  self.batch_size    = config.batch_size    or 1 --25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 150
  self.task          = config.task          or 'sic'  -- or 'vid'
  self.modelf        = config.modelf        or 'orig'  -- or 'vid'
	
  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)
  -- number of similarity rating classes
  if self.task == 'sic' then
    self.num_classes = 5
  elseif self.task == 'vid' then
    self.num_classes = 6
  elseif self.task == 'vid2' then
    self.num_classes = 6
  elseif self.task == 'msp' then
    self.num_classes = 2
  elseif self.task == 'quo' then
    self.num_classes = 2
  else
    error("not possible task!")
  end
	
  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  self.criterion = nn.DistKLDivCriterion()
  
  ----------------------------------------Combination of ConvNets.
  --
  if self.modelf == 'orig' then
    dofile 'models.lua'
  elseif self.modelf == 're' then -- more conv
    dofile 'models_re.lua'
  elseif self.modelf == 're1' then -- dkmax
    dofile 'models_re1.lua'
  elseif self.modelf == 're2' then --dkmax + more conv
    dofile 'models_re2.lua'
  elseif self.modelf == 're1ao' then --dkmax + more conv
    dofile 'models_re1ao.lua'
  elseif self.modelf == 're1a' then --dkmax + more conv
    dofile 'models_re1a.lua'
  elseif self.modelf == 're1aoa' then --dkmax + more conv
    dofile 'models_re1aoa.lua'
  else
      error("select models.lua file for create model")
  end
  print('<model> creating a fresh model')
  
  -- Type of model; Size of vocabulary; Number of output classes
  local modelName = 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt'
  print(modelName)
  self.ngram = 3
  self.length = self.emb_dim
  self.convModel = createModel(modelName, 10000, self.length, self.num_classes, self.ngram)  
  self.softMaxC = self:ClassifierOOne()

  ----------------------------------------
  local modules = nn.Parallel()
    :add(self.convModel) 
    :add(self.softMaxC) 
  self.params, self.grad_params = modules:getParameters()
  --print(self.params:norm())
  --print(self.convModel:parameters()[1][1]:norm())
  --print(self.softMaxC:parameters()[1][1]:norm())
end

function Conv:ClassifierOOne()
  local maxMinMean = 3
  local separator = (maxMinMean+1)*self.mem_dim
  local modelQ1 = nn.Sequential()	
  local ngram = self.ngram
  local items = (ngram+1)*3  		
  --local items = (ngram+1) -- no Min and Mean
  local NumFilter = self.length --300
  local conceptFNum = 20	
  inputNum = 2*items*items/3+NumFilter*items*items/3+6*NumFilter+(2+NumFilter)*2*ngram*conceptFNum --PoinPercpt model!
  modelQ1:add(nn.Linear(inputNum, self.sim_nhidden))
  modelQ1:add(nn.Tanh())	
  modelQ1:add(nn.Linear(self.sim_nhidden, self.num_classes))
  modelQ1:add(nn.LogSoftMax())	
  return modelQ1
end

function Conv:trainCombineOnly(dataset)
  --local classes = {1,2}
  --local confusion = optim.ConfusionMatrix(classes)
  --confusion:zero()
  train_looss = 0.0
   
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    if i%10 == 1 then
    	    xlua.progress(i, dataset.size)
    end

    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    -- get target distributions for batch
    local targets = torch.zeros(batch_size, self.num_classes)
    for j = 1, batch_size do
      local sim  = -0.1
      if self.task == 'sic' or self.task == 'vid' or self.task == 'vid2' or self.task=='msp' or self.task == 'quo' then
        sim = dataset.labels[indices[i + j - 1]] * (self.num_classes - 1) + 1
      elseif self.task == 'others' then
        sim = dataset.labels[indices[i + j - 1]] + 1 
      else
	error("not possible!")
      end
      local ceil, floor = math.ceil(sim), math.floor(sim)
      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil - sim
        targets[{j, ceil}] = sim - floor
      end--]]
    end
    
    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = self.emb_vecs:index(1, lsent:long()):double()
        local rinputs = self.emb_vecs:index(1, rsent:long()):double()
    		
   	    local part2 = self.convModel:forward({linputs, rinputs})
   	    local output = self.softMaxC:forward(part2)

        loss = self.criterion:forward(output, targets[1])
        train_looss = loss + train_looss
        local sim_grad = self.criterion:backward(output, targets[1])
		local gErrorFromClassifier = self.softMaxC:backward(part2, sim_grad)
		self.convModel:backward({linputs, rinputs}, gErrorFromClassifier)
      end
      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end
    _, fs  = optim.sgd(feval, self.params, self.optim_state)
    --train_looss = train_looss + fs[#fs]
  end
  print('Loss: ' .. train_looss..'\n')
  --xlua.progress(i , dataset.size)
end

-- Predict the similarity of a sentence pair.
function Conv:predictCombination(lsent, rsent)
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()

  local part2 = self.convModel:forward({linputs, rinputs})
  local output = self.softMaxC:forward(part2)
  local val = -1.0
  if self.task == 'sic' then
    val = torch.range(1, 5, 1):dot(output:exp())
  elseif self.task == 'vid' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'vid2' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'msp' then
    val = torch.range(0, 1, 1):dot(output:exp())
  elseif self.task == 'quo' then
    val = torch.range(0, 1, 1):dot(output:exp())
  else
    error("not possible task")
  end
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Conv:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    --local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent)
    if i%10 == 1 then
        xlua.progress(i,dataset.size)
    end
  end
  return predictions
end

function Conv:print_config()
  local num_params = self.params:nElement()
  print('Vocab length: ' .. self.emb_vecs:size(1))
  print('num params: ' .. num_params)
  print('word vector dim: ' .. self.emb_dim)
  print('LSTM memory dim: ' .. self.mem_dim)
  print('regularization strength: ' .. self.reg)
  print('minibatch size: ' .. self.batch_size)
  print('learning rate: ' .. self.learning_rate)
  print('LSTM structure: ' .. self.structure)
  print('LSTM layers: ' .. self.num_layers)
  print('sim module hidden dim: ' .. self.sim_nhidden)
end

function Conv:save(path)
  torch.save(path, self,'ascii')
end
