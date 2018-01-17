require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
--require 'cudnn'
--require 'cunn'   

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('Conv_cuda.lua')
--include('Conv.lua')
include('CsDis.lua')

--include('PaddingReshape.lua')
printf = utils.printf

-- global paths (modify if desired)
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end


modelTrained = torch.load("model/modelSTS.trained.th", 'ascii')
model=modelTrained
model:print_config()

--model.convModel:cuda()
-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')
vocab:add_unk_token()
-- load embeddings
--print(model.emb_vecs)

local taskD = 'sic'
local dev_dir = data_dir .. 'dev/'
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
--dev_dataset:cuda()
local dev_predictions = model:predict_dataset(dev_dataset)
local dev_score = pearson(dev_predictions, dev_dataset.labels)
printf('-- score: %.5f\n', dev_score)



--[[
modelTrained.convModel:evaluate()
modelTrained.softMaxC:evaluate()
local linputs = torch.zeros(rigth_sentence_length, emd_dimension)
linpus = XassignEmbeddingValuesX
local rinputs = torch.zeros(left_sentence_length, emd_dimension)
rinpus = XassignEmbeddingValuesX

local part2 = modelTrained.convModel:forward({linputs, rinputs})
local output = modelTrained.softMaxC:forward(part2)
local val = torch.range(0, 5, 1):dot(output:exp()) 
return val/5
]]
