require 'nn'
dofile 'models.lua'
dofile 'CsDis.lua'


ip1 = torch.rand(12,300)
ip2 = torch.rand(9,300)

mm = createModel('deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt', 10000, 300, 6, 3)
oo = mm:forward({ip1,ip2})
print(oo:size())
