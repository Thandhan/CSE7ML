import pandas as pd
dataset = pd.read_csv('2.csv')
concepts = dataset.iloc[:, :-1].values
target = dataset.iloc[:,-1].values
def learn(concepts, target):
    specific_h=concepts[0]
    print("Initialization of Specific_h and General_h")
    print(specific_h)
    general_h = [["?" for i in range(len(specific_h))]for i in range(len(specific_h))]
    print(general_h)
    for i,h in enumerate(concepts):
        if target[i].upper()=="YES":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    specific_h[x]="?"
                    general_h[x][x]="?"
        if target[i].upper()=="NO":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]="?"
        print("Steps of candidate elimination",i+1)
        print(specific_h)
        print(general_h)
    for i in general_h[:] :
        if i == ['?', '?', '?', '?', '?', '?']:
            general_h.remove(i)
    return specific_h,general_h
s_final, g_final=learn(concepts, target)
print("Final Specific_h:",s_final)
print("Final general_h:",g_final)