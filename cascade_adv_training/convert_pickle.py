import pickle
import os

y=[]
dirs = os.listdir('./')
for x in dirs:
    if 'perturb' in x and ('cascade' not in x and 'thermo' not in x):
        y.append(x)
        files = os.listdir(x)
        for z in files:
           temp = pickle.load(open(x + '/' +z,'rb'))
           pickle.dump(temp, open(x + '/' +z,'wb'), protocol=2)
