import os
import xml.etree.ElementTree as ET

filePath = "./2007"
count = 0

def dive(root):
	n = set()
	for child in root: 
		n.add(child.tag)
		x = dive(child)
		n = n.union(x)
	return n

def findTag(root,t):
	if(root.tag==t):
		return root
	for child in root:
		ft = findTag(child,t)
		if(ft): return ft
	return False

def findClass(root,c):
	if('class' in root.attrib):
		if(root.attrib['class']==c): return root
	for child in root:
		fc = findClass(child,c)
		if(fc): return fc
	return False

def text(root):
	txt = root.text.strip()
	if(txt): return txt
	for child in root: txt+= text(child)
	return txt

def dispText(root,name):
	print("___________")
	print(name)
	print(text(root))
	print("END")
	print("___________")

for month in os.listdir(filePath):
	filePath = "./2007/"+month
	for day in os.listdir(filePath):
		filePath = "./2007/"+month+"/"+day
		for filename in os.listdir(filePath):
			doc = ET.parse(filePath+"/"+filename)
			root = doc.getroot()
			abstract = findTag(root,'abstract')
			fullText = findClass(root,'full_text')
			if(abstract): dispText(abstract,"ABSTRACT")
			dispText(fullText,"FULL_TEXT")
			count+=1
			if(count>3): break
		break
	break
