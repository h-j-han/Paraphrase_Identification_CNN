import csv

def extract_into_list(file):
	f = open(file, 'r')

	list_return = []
	for line in f:
		list_return.append(float(line))

	return list_return

list_result = [extract_into_list("quo-results-dependency.1l.150d.epoch-.test_" + str(i+1) + ".2007.pred") for i in range(4)]
len_result = sum([len(list_result[i]) for i in range(4)])

list_final = [None for i in range(len_result)]

for i in range(len(list_final)):
	list_final[i] = list_result[(i+1) % 4][i // 4]

with open('test_result.csv', 'w') as csvfile:
	test_writer = csv.writer(csvfile)
	test_writer.writerow(['test_id', 'prediction_result'])
	for i in range(len(list_final)):
		test_writer.writerow([str(i), str(list_final[i])])