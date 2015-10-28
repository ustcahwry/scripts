import urllib
file=open('/home/ruoyu/code/TT/15 mismatch repair.txt')
data=[]
line=file.readline()
line=line.rstrip()
data.append(line)
while line!='':
    line=file.readline()
    line=line.rstrip()
    data.append(line)
file.close()
for str in data:
    url='http://tfgd.ihb.ac.cn/partial/expfig/ctg/locus/id/'+str
    print str
    path='/home/ruoyu/code/TT/'+str
    d=urllib.urlopen(url).read()
    f=open(path,"wb")
    f.write(d)
    f.close()

