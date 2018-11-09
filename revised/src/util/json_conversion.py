#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supports conversion between json and list of list of list of strings
"""
import json

## Given the full filename, deserialize and convert json object to documents 
## in the form of list of list of list of strings
def loadJsonToLslsls(filename):
    with open(filename, 'r') as infile:
        dic = json.load(infile)
    lslsls = [x for x in dic.values()]
    return lslsls

## Serialize the list of list of list of strings with assigned document id 
## and save into a json file with the given filename ("__.json")
def writeLslslsToJson(lslsls, filename):
    dic = {}
    for doc, i in zip(lslsls, range(len(lslsls))):
        dic[i] = doc   
    with open(filename, 'w') as outfile:
        json.dump(dic, outfile)
    return