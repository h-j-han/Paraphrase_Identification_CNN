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

-- read command line arguments
args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default dependency) Model architecture: [dependency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
  -b,--batch  (default 1)          Batch size
  -t,--task   (default vid2)       TaskD vid2 for msrvid2 and quo for QUORA
  -r,--thread (default 4)          number of torch.setnumthreads( )
  -o,--option (default train)      train or test option
  -x,--loadDir (default modelSTS.trained.th)  Loaded model for testing
  -s,--save (default false)        Save the train model
]]
-- layers : 1
-- model : "dependency:
-- dim : 150

--local model_name, model_class, model_structure
model_name = 'convOnly'
model_class = similarityMeasure.Conv
model_structure = model_name

torch.seed()

print('<torch> using the specified seed: ' .. torch.initialSeed())
--local data_dir = 'data/msrvid2/' 
local data_dir
if args.task == 'vid' then
  data_dir = 'data/msrvid/' 
elseif args.task == 'quo' then
  data_dir = 'data/quora/' 
else 
  print('f\n')
end

vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')
print('loading word embeddings')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
emb_dim = emb_vecs:size(2)
-- emb_vecs:size()
-- 2196017
--     300
-- [torch.LongStorage of size 2]

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
--local taskD = 'vid2'
local taskD = args.task
-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD)
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)
printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model -- Conv:__init(config)   
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  mem_dim    = args.dim,
  num_layers = args.layers,
  task       = taskD,
  batch_size = args.batch
}

-- number of epochs to train
local num_epochs = 35

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()


if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
torch.setnumthreads(args.thread)
print('<torch> number of threads in used: ' .. torch.getnumthreads())

header('Training model')

local id = 2007
print("Id: " .. id)
if args.option=='train' then
  for i = 1, num_epochs do
    local start = sys.clock()
    print('--------------- EPOCH ' .. i .. '--- -------------')
    model:trainCombineOnly(train_dataset)
    print('Finished epoch in ' .. ( sys.clock() - start) )

    local dev_predictions = model:predict_dataset(dev_dataset)
    local dev_score = pearson(dev_predictions, dev_dataset.labels)
    printf('-- dev score: %.5f\n', dev_score)

    --write prediction
    if dev_score >= best_dev_score then
      best_dev_score = dev_score
      local test_predictions = dev_predictions--local test_predictions = model:predict_dataset(test_dataset)
      local test_score = pearson(test_predictions, test_dataset.labels)
      printf('[[BEST DEV]]-- dev score: %.4f\n [[ITS TEST]]-- test score: %.4f\n', dev_score,test_score)

      local predictions_save_path = string.format(
          similarityMeasure.predictions_dir .. '/%sresults-%s.%dl.%dd.epoch-%.2d.%.3f.%d.pred',taskD,args.model, args.layers, args.dim, i, dev_score, id)
      local predictions_file = torch.DiskFile(predictions_save_path, 'w')
      print('writing predictions to ' .. predictions_save_path)
      for i = 1, test_predictions:size(1) do
        predictions_file:writeFloat(test_predictions[i])
        --if i%10 == 1 then
            xlua.progress(i,test_predictions:size(1))
        --end
      end
      predictions_file:close()
    end
    if args.save==true then
      print('saving...')
      model:save(string.format('./model/%smodel.epoch%d',taskD,i))
    end
  end
  print('finished training in ' .. (sys.clock() - train_start))
elseif args.option=='test' then
  print('Test mode: '..args.loadDir)
  loadDir='model/'..args.loadDir
  model = torch.load(loadDir, 'ascii')
  model:print_config()
  local test_predictions = model:predict_dataset(test_dataset)
  local test_score = pearson(test_predictions, test_dataset.labels)
  printf('-- score: %.5f\n', test_score)
  
  local predictions_save_path = string.format(
      similarityMeasure.predictions_dir .. '/%sresults-%s.%dl.%dd.epoch-.%.3f.%d.pred',taskD,args.model, args.layers, args.dim, test_score, id)
  local predictions_file = torch.DiskFile(predictions_save_path, 'w')
  print('writing predictions to ' .. predictions_save_path)
  for i = 1, test_predictions:size(1) do
    predictions_file:writeFloat(test_predictions[i])
    --if i%10 == 1 then
        xlua.progress(i,test_predictions:size(1))
    --end
  end
  predictions_file:close()
  
  
else
  print('Wrong option input')
end











