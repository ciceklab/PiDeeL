import numpy as np 
import matplotlib.pyplot as plt
import pdb
import pickle
# load ages
with open('/home/gunkaynar/survival_analysis/targeted/age_no_imputation.pickle', 'rb') as handle:
    ages = pickle.load(handle)
b = []
for i in range(len(ages)):
    if ages[i] == "remove":
        b.append(i)


ages = np.delete(ages,b)
ages = ages.astype(np.int32)
plt.figure()
plt.xlabel("Age (Year)")
plt.ylabel("Patient count")
plt.hist(ages, bins=[10,20,30,40,50,60,70,80,90])
plt.savefig("./age_hist.v2.pdf")

print(np.mean(ages))
print(np.min(ages))
print(np.max(ages))
print(np.std(ages))