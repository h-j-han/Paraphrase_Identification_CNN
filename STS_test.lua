require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('Conv.lua')
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


-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')
vocab:add_unk_token()
-- load embeddings
--[[
print('loading word embeddings')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')

local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()
]]

--print(model.emb_vecs)

local taskD = 'sic'
local dev_dir = data_dir .. 'dev/'
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)

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