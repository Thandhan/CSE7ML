import pandas as pd
import math
import random
import numpy as np
def majorclass(attributes,data,target):
    freq={}
    index=attributes.index(target)
    for tuple in data:
        if tuple[index] in freq:
            freq[tuple[index]]+=1
        else:
            freq[tuple[index]]=1
    max=0
    major=""
    for key in freq.keys():
        if freq[key]>max:
            max=freq[key]
            major=key
    return major

def entropy(attributes,data,targetAttr):
    freq={}
    dataEntropy=0.0
    i=0
    for entry in attributes:
        if(targetAttr==entry):
            break
        i=i+1
    i=i-1
    for entry in data:
        if entry[i] in freq:
            freq[entry[i]]+=1.0
        else:
            freq[entry[i]]=1.0
    for freq in freq.values():
        dataEntropy+=(-freq/len(data))*math.log(freq/len(data),2)
    return dataEntropy
def info_gain(attributes,data,attr,targetAttr):
    freq={}
    subsetEntropy=0.0
    i=attributes.index(attr)
    for entry in data:
        if entry[i] in freq:
            freq[entry[i]]+=1.0
        else:
            freq[entry[i]]=1.0
    for val in freq.keys():
        valProb=freq[val]/sum(freq.values())
        datasubset=[entry for entry in data if entry[i]==val]
        subsetEntropy+=valProb*entropy(attributes,datasubset,targetAttr)
    return(entropy(attributes,data,targetAttr)-subsetEntropy)
def attr_choose(data,attributes,target):
    best=attributes[0]
    maxGain=0
    for attr in attributes:
        newGain =info_gain(attributes,data,attr,target)
        if newGain>maxGain:
            maxGain=newGain
            best=attr
    return best
def get_values(data,attributes,attr):
    index=attributes.index(attr)
    values=[]
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values
def get_data(data,attributes,best,val):
    new_data=[[]]
    index=attributes.index(best)
    for entry in data:
        if(entry[index]==val):
            newEntry=[]
            for i in range(0,len(entry)):
                if(i!=index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)
    new_data.remove([])
    return new_data

def build_tree(data,attributes,target):
    data=data[:]
    vals=[record[attributes.index(target)] for record in data]
    default=majorclass(attributes,data,target)
    if not data or (len(attributes)-1)<=0:
        return default
    elif vals.count(vals[0])==len(vals):
        return vals[0]
    else:
        best=attr_choose(data,attributes,target)
        tree={best:{}}
        for val in get_values(data,attributes,best):
            new_data=get_data(data,attributes,best,val)
            newAttr=attributes[:]
            newAttr.remove(best)
            subtree=build_tree(new_data,newAttr,target)
            tree[best][val]=subtree
    return tree
    
def execute_decision_tree():
    data = pd.read_csv('3.csv')
    data = data.iloc[:, :].values
    print("number of records:",len(data))
    attributes = ['outlook','temperature','humidity','wind','play']
    target = attributes[-1]
    training_set = [np.array(i) for i in data]
    tree = build_tree(training_set,attributes,target)
    print(tree)
    test_set=[('sunny','hot','high','weak')]
    for entry in test_set:
        tempDict = tree.copy()
        result=""
        while(isinstance(tempDict, dict)):
            nodeVal = next(iter(tempDict))
            tempDict = tempDict[next(iter(tempDict))]
            index = attributes.index(nodeVal)
            value = entry[index]
            if(value in tempDict.keys()):
                result = tempDict[value]
                tempDict = tempDict[value]
            else:
                result="Null"
                break
    print(result)
execute_decision_tree()
