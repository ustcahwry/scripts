# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:29:33 2015

@author: ruoyu
"""

import bs4
import requests
def Goannotations(soup):
    gostr=[]
    i=0
    for string in soup.stripped_strings:
      if string == u'Domains':
          break
      if i==1:
          gostr.append(string)
      if string == u'Gene Ontology Annotations':
          i=1
    return gostr
def GeneName(soup):
    namestr=[]
    i=0
    for string in soup.stripped_strings:
      if string == u'Aliases':
          break
      if i==1:
          namestr.append(string)
      if string == u'Standard Name':
          i=1
    return namestr
def Description(soup):
    dscstr=[]
    i=0
    for string in soup.stripped_strings:
      if string == u'Genome Browser (Macronucleus)':
          break
      if i==1:
          dscstr.append(string)
      if string == u'Description':
          i=1
    return dscstr

file = open('2hr-sig.csv')
data = []
line = file.readline()
line=line.rstrip()
data.append(line)
while line!='':
    line = file.readline()
    line=line.rstrip()
    data.append(line)
fl=open('Annot-2hr.txt', 'w')
n=0
for str in data:
      response = requests.get('http://ciliate.org/index.php/feature/details/'+str)
      soup= bs4.BeautifulSoup(response.text)
      n=n+1
      print str
      print n
      str1=GeneName(soup)
      str2=Description(soup)
      str3=Goannotations(soup)
      fl.write(repr(str))
      fl.write("\t")
      for c in str1:
         c=c.encode('utf8')
         fl.write(repr(c))
      fl.write("\t")
      for c in str2:
         c=c.encode('utf8')
         fl.write(repr(c))
      fl.write("\t")
      for c in str3:
         c=c.encode('utf8')
         fl.write(repr(c))
      fl.write("\t")
      fl.write("\n")

fl.close()

