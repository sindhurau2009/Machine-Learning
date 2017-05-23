import xlrd
from xlrd.sheet import ctype_text
import math
import numpy
import scipy
from scipy import stats
from scipy.stats import norm
#import matplotlib
#import matplotlib.pyplot as plt

xl_workbook = xlrd.open_workbook("university data.xlsx")

sheet_names = xl_workbook.sheet_names()

xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])

print("UBitName: suppu")
print("personNumber: 50206730")


row_len = len(xl_sheet.col_values(0))

s1 = 0
x1 = 0
for i in range(1,50):
    val = xl_sheet.cell(i,2).value
    s1 = s1 + val
    x1 = x1 + 1
mu1 = s1 * 1 / x1
mu1 = round(mu1,3)
print("mu1: "+str(mu1))

s2 = 0
x2 = 0
for i in range(1,50):
    val = xl_sheet.cell(i,3).value
    s2 = s2 + val
    x2 = x2 + 1
mu2 = s2 * 1 / x2
mu2 = round(mu2,3)
print("mu2: "+str(mu2))

s3 = 0
x3 = 0
for i in range(1,50):
    val = xl_sheet.cell(i,4).value
    s3 = s3 + val
    x3 = x3 + 1
mu3 = s3 * 1 / x3
mu3 = round(mu3,3)
print("mu3: "+str(mu3))

s4 = 0
x4 = 0
for i in range(1,50):
    val = xl_sheet.cell(i,5).value
    s4 = s4 + val
    x4 = x4 + 1
mu4 = s4 * 1 / x4
mu4 = round(mu4,3)
print("mu4: "+str(mu4))

s1 = 0
x1 = 0
for i in range(1,50):
    val = (xl_sheet.cell(i,2).value - mu1) * (xl_sheet.cell(i,2).value - mu1)
    s1 = s1 + val
    x1 = x1 + 1
var1 = s1 * 1 / (x1 - 1)
var1 = round(var1,3)
print("var1: "+str(var1))

s2 = 0
x2 = 0
for i in range(1,50):
    val =(xl_sheet.cell(i,3).value - mu2) * (xl_sheet.cell(i,3).value - mu2)
    s2 = s2 + val
    x2 = x2 + 1
var2 = s2 * 1 / x2
var2 = round(var2,3)
print("var2: "+str(var2))

s3 = 0
x3 = 0
for i in range(1,50):
    val =(xl_sheet.cell(i,4).value - mu3) * (xl_sheet.cell(i,4).value - mu3)
    s3 = s3 + val
    x3 = x3 + 1
var3 = s3 * 1 / x3
var3 = round(var3,3)
print("var3: "+str(var3))

s4 = 0
x4 = 0
for i in range(1,50):
    val =(xl_sheet.cell(i,5).value - mu4) * (xl_sheet.cell(i,5).value - mu4)
    s4 = s4 + val
    x4 = x4 + 1
var4 = s4 * 1 / x4
var4 = round(var4,3)
print("var4: "+str(var4))

sigma1 = math.sqrt(var1)
sigma1 = round(sigma1,3)
print("sigma1: "+str(sigma1))
sigma2 = math.sqrt(var2)
sigma2 = round(sigma2,3)
print("sigma2: "+str(sigma2))
sigma3 = math.sqrt(var3)
sigma3 = round(sigma3,3)
print("sigma3: "+str(sigma3))
sigma4 = math.sqrt(var4)
sigma4 = round(sigma4,3)
print("sigma4: "+str(sigma4))


l1 = []
l2 = []
l3 = []
l4 = []

for i in range(1,50):
  t1 = xl_sheet.cell(i,2).value
  l1.append(t1)
  t2 = xl_sheet.cell(i,3).value
  l2.append(t2)
  t3 = xl_sheet.cell(i,4).value
  l3.append(t3)
  t4 = xl_sheet.cell(i,5).value
  l4.append(t4)
  

csscorearr = numpy.array(l1)
researcharr = numpy.array(l2)
basepayarr = numpy.array(l3)
tuitionarr = numpy.array(l4)

covar = numpy.cov(numpy.vstack((csscorearr,researcharr,basepayarr,tuitionarr)))
print("covarianceMat:")
print(str(covar))

correlation = numpy.corrcoef(numpy.vstack((csscorearr,researcharr,basepayarr,tuitionarr)))
print("correlationMat:")
print(str(correlation))

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(l1,l2)
#plt.xlabel('CS Score (USNews)')
#plt.ylabel('Research Overhead')
#plt.savefig('images/l1l2.jpeg')

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(l1,l3)
#plt.xlabel('CS Score(USNews)')
#plt.ylabel('Admin Base Pay')
#plt.savefig('images/l1l3.jpeg')

#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#ax3.plot(l1,l4)
#plt.xlabel('CS Score (USNews)')
#plt.ylabel('Tuition((out-state)')
#plt.savefig('images/l1l4.jpeg')

#fig4 = plt.figure()
#ax4 = fig4.add_subplot(111)
#ax4.plot(l2,l3)
#plt.xlabel('Research Overhead')
#plt.ylabel('Admin Base Pay')
#plt.savefig('images/l2l3.jpeg')

#fig5 = plt.figure()
#ax5 = fig5.add_subplot(111)
#ax5.plot(l2,l4)
#plt.xlabel('Research Overhead')
#plt.ylabel('Tuition(out-state)')
#plt.savefig('images/l2l4.jpeg')

#fig6 = plt.figure()
#ax6 = fig6.add_subplot(111)
#ax6.plot(l3,l4)
#plt.xlabel('Admin Base Pay')
#plt.ylabel('Tuition(out-state)')
#plt.savefig('images/l3l4.jpeg')


pdf1 = norm(mu1,sigma1).logpdf(l1)
pdf1 = sum(pdf1)

pdf2 = norm(mu2,sigma2).logpdf(l2)
pdf2 = sum(pdf2)

pdf3 = norm(mu3,sigma3).logpdf(l3)
pdf3 = sum(pdf3)

pdf4 = norm(mu4,sigma4).logpdf(l4)
pdf4 = sum(pdf4)

pdf5 = pdf1 + pdf2 + pdf3 + pdf4

LogLikelihood = [pdf1,pdf2,pdf3,pdf4]
pdf5 = round(pdf5,3)
print("LogLikelihood: "+str(pdf5))

mat = [[0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
BNGraph = numpy.matrix(mat)
print("BNgraph:")
print(BNGraph)

dat = numpy.zeros(csscorearr.size)
for i in range(dat.size):
    dat[i] = 1.0
data = [dat,researcharr,basepayarr,tuitionarr]
coeffmat = numpy.zeros((4,4))
for i in range(4):
    for j in range(4):
        sum1 = 0
        for k in range(csscorearr.size):
            sum1 = sum1 + (data[i][k]*data[j][k])
        coeffmat[i][j] = sum1
    
yfunmat = numpy.zeros((4,1))
for i in range(4):
    sum1 = 0
    for j in range(csscorearr.size):
        sum1 = sum1 + (data[i][j] * csscorearr[j])
    yfunmat[i] = sum1


betaarr = numpy.linalg.solve(coeffmat,yfunmat)

numsum = 0
for i in range(csscorearr.size):
    sum2 = 0
    for j in range(4):
        sum2 = sum2 + (data[j][i]*betaarr[j][0])
    sum2 = sum2 - csscorearr[i]
    numsum = numsum + (sum2 * sum2)
stddev = numpy.sqrt(numsum /csscorearr.size)
var = numsum / csscorearr.size


dt = numpy.array("data")



loghd = -(0.5 * numpy.log(2 * numpy.pi * var))

BNLogHd = 0
for i in range(csscorearr.size):
    sum1 = 0
    BNLogHd = BNLogHd + loghd
    for j in range(len(data)):
        sum1 = sum1 + (data[j][i] * betaarr[j][0])
    sum1 = sum1 - csscorearr[i]
    sum2 = (-(sum1 * sum1) / (2 * var))
    BNLogHd = BNLogHd + sum2

i = len(LogLikelihood)
if i!=0:
    j = 1
    while j<i:
        BNLogHd = BNLogHd + LogLikelihood[j]
        j = j+1
BNLogHd = round(BNLogHd,3)
print("BNlogLikelihood: "+str(BNLogHd))    
