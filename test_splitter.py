"""
This file is being used for splitting the test file into
4 pieces, each with the almost equal number of data
"""

file1_splitted = "./a.toks"
file2_splitted = "./b.toks"
id_splitted = "./id.txt"

f1 = open(file1_splitted, "r")
f2 = open(file2_splitted, "r")
id_s = open(id_splitted, "r")

target = [open("./test_" + str(i+1) + "/a.toks", "w") for i in range(4)]
target_b = [open("./test_" + str(i+1) + "/b.toks", "w") for i in range(4)]
id_x = [open("./test_" + str(i+1) + "/id.txt", "w") for i in range(4)]

count_data = 0

#handling the data for the a.toks
for line in f1:
	count_data += 1
	target[count_data % 4].write(line)
count_data = 0

#handling the data for the b.toks
for line in f2:
	count_data += 1
	target_b[count_data % 4].write(line)
count_data = 0

#handling the data for the id.txt
for line in id_s:
	count_data += 1
	id_x[count_data % 4].write(line)
count_data = 0
