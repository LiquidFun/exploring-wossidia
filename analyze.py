import csv
from collections import Counter

with open("ISEBEL-Datasets/witches-edges.csv") as file:
    lines = file.readlines()
    counter = Counter()
    for line in lines:
        id1, id2, *_ = line.split(',')
        counter[id1] += 1
        counter[id2] += 1
print(*counter.most_common(100))
