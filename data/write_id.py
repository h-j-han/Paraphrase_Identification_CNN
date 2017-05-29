import os

train = 'train'
test = 'test'
base = './quora/'
idtxt = 'id.txt'
simtxt = 'sim.txt'
atok = 'a.toks'
with open(base+train+'/'+simtxt) as f:
    train_num = sum(1 for line in f)
print train_num

with open(base+test+'/'+atok) as f:
    test_num = sum(1 for line in f)
print test_num

with open(base+train+'/'+idtxt, 'w') as f1:
    for i in xrange(train_num):
        f1.write(str(i) + os.linesep)

with open(base+test+'/'+idtxt, 'w') as f2:
    for i in xrange(test_num):
        f2.write(str(i+train_num) + os.linesep)

