import os
import io
train = 'train'
test = 'test'
base = './msrp/'
idtxt = 'id.txt'
simtxt = 'sim.txt'
simtxtc = 'sim.txt.bak'

a = []
bb = 0
with io.open(base+train+'/'+simtxtc,"r",encoding='utf-8-sig' ) as f:
    for line in f:
        #c = int(line.split()[0])
        #print(c)
        #if not bb :
        #    aa = line
        #if bb :
        a.append(int(line))
        #bb +=1

with open(base+train+'/'+simtxt, 'w') as f1:
    for i in a:

        f1.write(str(i)+'.0' + os.linesep)
'''
with open(base+test+'/'+simtxt) as f:
    test_num = sum(1 for line in f)
print test_num

with open(base+train+'/'+idtxt, 'w') as f1:
    for i in xrange(train_num):
        f1.write(str(i) + os.linesep)

with open(base+test+'/'+idtxt, 'w') as f2:
    for i in xrange(test_num):
        f2.write(str(i+train_num) + os.linesep)
'''
