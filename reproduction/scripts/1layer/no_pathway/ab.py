import sys
sys.path.insert(1, "../../")
from load_targeted_data import predicted_quant, events, survival, tumor_grade


malignancy = [0 if i == "1" or i == "2" else 1 for i in tumor_grade]
#print low survival rate samples whose tumor grade is 1 or 2
for i in range(len(survival)):
    if malignancy[i] == 0 and survival[i] < 100:
        print(i, survival[i], malignancy[i])
    if malignancy[i]== 1 and survival[i]>5000:
        print(i, survival[i], malignancy[i])