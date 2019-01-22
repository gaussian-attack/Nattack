import matplotlib.pyplot as plt
import numpy as np
import itertools

target_names = ['THERM','THERM-ADV','LID','SAP','RSE','CAS-ADV','ORIGIN','WRES-28']
cm = np.array([[100.00 ,2.90 , 3.27 , 5.38 , 7.67 , 1.80 , 5.30,11.06],
 [2.01 , 100 , 1.11 , 1.47 , 1.73, 1.42 , 1.69, 1.9],
[10.77 , 10.34 , 100 ,  6.68, 11.27, 6.18,28.58, 15.66],
[20.00 , 6.13 , 4.14 , 100 , 17.98 , 4.70 , 10.72,26.63],
[ 15.85 , 3.74 , 4.84 , 8.51 , 100 , 2.28 , 10.83,18.11],
[3.70 , 7.42 , 1.67 , 3.83 , 2.76 , 100 , 11.66, 12.88],
[ 9.14 , 6.16 ,15.64 ,4.89 , 10.29 , 4.69, 100.00, 12.49],
[26.51,4.95,7.05,14.75,20.26,4.15,14.59,100.0]])

# target_names = ['SAP','THERM-ADV','THERM','LID','ORIGIN','WRES-28']
# cm = np.array([[100,19.64,80.02	,30.54,	2.12,	8.51],
# 	[4.13,	100,	2.15,	3.14,	1.23,	1.13],
# 	[67.11,	20.02	,100	,52.66,	1.34	,2.17],
# 	[30.1,	20.83,	34.26,	100,	8.92,	3.15],
# 	[22.15,	12.75,	25.16,	48.51,	100,	0.21],
# 	[97.62,	17.1,	88.23,	37.93,	0.36	,100]])


#if cmap is None:
cmap = plt.get_cmap('tab20b') ###  RdGy cool tab20b tab20c

plt.figure(figsize=(8, 6.5))
#plt.subplot(1,2,1)

plt.imshow(cm, interpolation='nearest', cmap=cmap)
#plt.title('Confusion Matrix')
plt.colorbar()

if target_names is not None:
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

#if normalize:
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


thresh = cm.max() / 100

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize= 12)
    # plt.text(j, i, "{:,}".format(cm[i, j]),
    #          horizontalalignment="center",
    #          color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
#plt.ylabel('Defense methods', fontsize = 16)
#plt.xlabel('Perturbation', fontsize = 16)

#plt.subplot(1,2,2)


plt.show()
