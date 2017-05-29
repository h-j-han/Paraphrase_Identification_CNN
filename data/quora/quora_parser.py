"""
This file is being used for the purpose of
parsing the corpus file of the MSRP file
"""

import os
import csv

sentence_file1 = []
sentence_file2 = []

file_test = "./test.csv"
file_train = "./train.csv"

file1_token_a = "./train_quora/a.toks"
file1_token_b = "./train_quora/b.toks"
file1_id = "./train_quora/id.txt"
file1_sim = "./train_quora/sim.txt"

file2_token_a = "./test_quora/a.toks"
file2_token_b = "./test_quora/b.toks"
file2_id = "./test_quora/id.txt"
file2_sim = "./test_quora/sim.txt"

f1_token_a = open(file1_token_a, 'w')
f1_token_b = open(file1_token_b, 'w')
f1_id = open(file1_id, 'w')
f1_sim = open(file1_sim, 'w')

sentence_a = ''
sentence_b = ''

# for i in range(len(sentence_file1)):
# 	for j in range(len(sentence_file1[i][3])):
# 		if (ord(sentence_file1[i][3][j]) > 64 and ord(sentence_file1[i][3][j]) < 91) or (ord(sentence_file1[i][3][j]) > 96 and ord(sentence_file1[i][3][j]) < 123) or ord(sentence_file1[i][3][j]) == 32\
# 			or (ord(sentence_file1[i][3][j]) > 47 and ord(sentence_file1[i][3][j]) < 58):
# 			sentence_a += sentence_file1[i][3][j]
# 		else:
# 			if ((sentence_file1[i][3][j] == '.' or sentence_file1[i][3][j] == ',') and j != len(sentence_file1[i][3]) - 1):
# 				if ord(sentence_file1[i][3][j+1]) in range(48, 58) and ord(sentence_file1[i][3][j-1]) in range(48,58):
# 					sentence_a += sentence_file1[i][3][j]
# 			else:
# 				if (sentence_file1[i][3][j] == "$" and ord(sentence_file1[i][3][j+1]) in range(48,58)):
# 					sentence_a += sentence_file1[i][3][j]
# 				else:
# 					sentence_a += " " + sentence_file1[i][3][j] + " "
# 	# print(sentence_a)
# 	for j in range(len(sentence_file1[i][4])):
# 		if (ord(sentence_file1[i][4][j]) > 64 and ord(sentence_file1[i][4][j]) < 91) or (ord(sentence_file1[i][4][j]) > 96 and ord(sentence_file1[i][4][j]) < 123) or ord(sentence_file1[i][4][j]) == 32\
# 			or (ord(sentence_file1[i][4][j]) > 47 and ord(sentence_file1[i][4][j]) < 58):
# 			sentence_b += sentence_file1[i][4][j]
# 		else:
# 			if ((sentence_file1[i][4][j] == '.' or sentence_file1[i][4][j] == ',') and j != len(sentence_file1[i][4]) - 1):
# 				if ord(sentence_file1[i][4][j+1]) in range(48, 58) and ord(sentence_file1[i][4][j-1]) in range(48,58):
# 					sentence_b += sentence_file1[i][4][j]
# 			else:
# 				if (sentence_file1[i][4][j] == "$" and ord(sentence_file1[i][4][j+1]) in range(48,58)):
# 					sentence_b += sentence_file1[i][4][j]
# 				else:
# 					sentence_b += " " + sentence_file1[i][4][j] + " "

#now processing the training file
with open(file_train, 'r') as f1:
	reader = csv.reader(f1)
	for row in reader:
		for j in range(len(row[3])):
			if (ord(row[3][j]) in range(65, 91) or ord(row[3][j]) in range(97, 123) or ord(row[3][j]) == 32 or ord(row[3][j]) in range(48,58)):
				sentence_a += row[3][j]
			else:
				if ((row[3][j] == '.' or row[3][j] == ',') and j != len(row[3]) - 1):
					if ord(row[3][j+1]) in range(48,58) and ord(row[3][j-1]) in range(48,58):
						sentence_a += row[3][j]
					else:
						if (row[3][j] != ' ' and ord(row[3][j+1]) in range(48,58)):
							sentence_a += row[3][j]
						else:
							sentence_a += " " + row[3][j] + " "
		for j in range(len(row[4])):
			if (ord(row[4][j]) in range(65, 91) or ord(row[4][j]) in range(97, 123) or ord(row[4][j]) == 32 or ord(row[4][j]) in range(48,58)):
				sentence_b += row[4][j]
			else:
				if ((row[4][j] == '.' or row[4][j] == ',') and j != len(row[4]) - 1):
					if ord(row[4][j+1]) in range(48,58) and ord(row[4][j-1]) in range(48,58):
						sentence_b += row[4][j]
					else:
						if (row[4][j] != ' ' and ord(row[4][j+1]) in range(48,58)):
							sentence_b += row[4][j]
						else:
							sentence_b += " " + row[4][j] + " "

		f1_token_a.write(sentence_a + "\n")
		f1_token_b.write(sentence_b + "\n")
		f1_id.write(row[1] + " " + row[2] + "\n")
		f1_sim.write(row[5] + "\n")

sentence_a = ''
sentence_b = ''
print("Training file updated")

f1_token_a.close()
f1_token_b.close()
f1_id.close()
f1_sim.close()

f2_token_a = open(file2_token_a, 'w')
f2_token_b = open(file2_token_b, 'w')
f2_id = open(file2_id, 'w')
# f2_sim = open(file2_sim, 'w')

with open(file_test, 'r') as f2:
	reader = csv.reader(f2)
	for row in reader:
		for j in range(len(row[1])):
			if (ord(row[1][j]) in range(65, 91) or ord(row[1][j]) in range(97, 123) or ord(row[1][j]) == 32 or ord(row[1][j]) in range(48,58)):
				sentence_a += row[1][j]
			else:
				if ((row[1][j] == '.' or row[1][j] == ',') and j != len(row[1]) - 1):
					if ord(row[1][j+1]) in range(48,58) and ord(row[1][j-1]) in range(48,58):
						sentence_a += row[1][j]
					else:
						if (row[1][j] != ' ' and ord(row[1][j+1]) in range(48,58)):
							sentence_a += row[1][j]
						else:
							sentence_a += " " + row[1][j] + " "
		for j in range(len(row[2])):
			if (ord(row[2][j]) in range(65, 91) or ord(row[2][j]) in range(97, 123) or ord(row[2][j]) == 32 or ord(row[2][j]) in range(48,58)):
				sentence_b += row[2][j]
			else:
				if ((row[2][j] == '.' or row[2][j] == ',') and j != len(row[2]) - 1):
					if ord(row[2][j+1]) in range(48,58) and ord(row[2][j-1]) in range(48,58):
						sentence_b += row[2][j]
					else:
						if (row[2][j] != ' ' and ord(row[2][j+1]) in range(48,58)):
							sentence_b += row[2][j]
						else:
							sentence_b += " " + row[2][j] + " "

		f2_token_a.write(sentence_a + "\n")
		f2_token_b.write(sentence_b + "\n")
		f2_id.write(row[0] + "\n")
		# f2_sim.write(row[5] + "\n")

print("Testing file updated")
f2_token_a.close()
f2_token_b.close()
f2_id.close()
# f2_sim.close()

# #next processing the testing file
# f2 = open(file2, "r")
# for line in f2:
# 	storage = line.split("\t")
# 	for i in range(len(storage)):
# 		storage[i] = storage[i].strip()
# 	sentence_file2.append(storage)
# f2.close()

# f_token_a = open(file1_token_a, "w")
# f_token_b = open(file1_token_b, "w")
# f_id = open(file1_id, "w")
# f_sim = open(file1_sim, "w")
# for i in range(len(sentence_file1)):
# 	f_token_a.write(sentence_file1[i][3] + "\n")
# 	f_token_b.write(sentence_file1[i][4] + "\n")
# 	f_sim.write(sentence_file1[i][0] + "\n")
# 	f_id.write(sentence_file1[i][1] + " " + sentence_file1[i][2] + "\n")
# f_token_a.close()
# f_token_b.close()
# f_sim.close()
# f_id.close()

# f2_token_a = open(file2_token_a, "w")
# f2_token_b = open(file2_token_b, "w")
# f2_id = open(file2_id, "w")
# f2_sim = open(file2_sim, "w")
# for i in range(len(sentence_file2)):
# 	f2_token_a.write(sentence_file2[i][3] + "\n")
# 	f2_token_b.write(sentence_file2[i][4] + "\n")
# 	f2_sim.write(sentence_file2[i][0] + "\n")
# 	f2_id.write(sentence_file2[i][1] + " " + sentence_file2[i][2] + "\n")
# f2_token_a.close()
# f2_token_b.close()
# f2_sim.close()
# f2_id.close()