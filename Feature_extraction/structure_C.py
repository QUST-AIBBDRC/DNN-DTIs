import numpy
import pandas as pd
import numpy as np

#result = [[0 for x in range(110)] for y in range(62)]
#p=0
#pathname='nr_protein_fasta/'
#zifu='s'+str(p)
#zifu2=pathname+zifu+'.spd33'
#file_read = open(zifu2)
#num_lines = sum(1 for line in open(zifu2))
#i=0
#a=[]
#while i < num_lines:
#      i += 1
#      str1 = file_read.readline()
#      l = str1.rstrip('\n').split()
#      H_=l[3:8]+l[10:]
#      H_H=list(H_)
#      a.append(H_H)
#      H_H=[]
#      H_=[]
#with open('test.txt') as f:
#    lines = f.readlines()[1:]
#    for line in lines:
#        print(line,end='')

result=[]
pathname='ic_protein_fasta/'

def read_spd_C(pathname): 
    for p in range(50):   
        zifu='s'+str(p)
        zifu2=pathname+zifu+'.spd33'
        file_read = open(zifu2)
        num_lines = sum(1 for line in open(zifu2))
        matrix = [[0 for x in range(8)] for y in range(10)]
        probability_matrix = [[0 for x in range(3)] for y in range(10)]
        i = 0
        a = []
        while i < num_lines:
              i += 1
              str1 = file_read.readline()
              l = str1.rstrip('\n').split()
              H_=l[3:8]+l[10:]
              H_H=list(H_)
              a.append(H_H)
              H_H=[]
              H_=[]
        a.pop(0)
        num_lines=len(a)
        array = a
        degree_matrix = [[0 for x in range(8)] for y in range(num_lines-1)]
        degree_index = 0
        for x in range(0,num_lines-1):
            degree_index = 0
            for y in range(1,5):
                degree_matrix[x][degree_index] = numpy.math.sin(float(array[x][y]) * numpy.pi / 180 )
                degree_matrix[x][degree_index+1] = numpy.math.cos(float(array[x][y]) * numpy.pi / 180 )
                degree_index += 2
        print(len(array))
        print(len(array[0]))
        print(len(degree_matrix))
        print(len(degree_matrix[0]))
        temp_array = array
        array = degree_matrix
        for k in range(0, 10):
            for j in range(0, 8):
                for i in range(0, num_lines - 1 - k):
                    matrix[k][j] += float(array[i][j]) * float(array[i + k][j])
                    matrix[k][j] = matrix[k][j] / (num_lines - 1)
        array = temp_array
        for k in range(0, 10):
            for j in range(5, 8):
                for i in range(0, num_lines - 1 - k):
                    probability_matrix[k][j - 5] += numpy.float(array[i][j]) * numpy.float(array[i + k][j])
                    probability_matrix[k][j - 5] = probability_matrix[k][j - 5] / (num_lines - 1)
        final_matrix = [[0 for x in range(110)] for y in range(1)]
        index = 0
        for m in range(0, len(matrix)):
            for n in range(0, len(matrix[m])):
                final_matrix[0][index] = matrix[m][n]
                index += 1
        for m in range(0, len(probability_matrix)):
            for n in range(0, len(probability_matrix[0])):
                final_matrix[0][index] = probability_matrix[m][n]
                index += 1
        matrix = np.array(final_matrix)        
        vector1=matrix.astype(float)
        #vector2=vector1.T
        #vector2=vector1[:,0]
        result.append(vector1)
    return result
result=read_spd_C(pathname)
result=np.array(result)
vector=np.reshape(result,(result.shape[0],result.shape[1]*result.shape[2]))
#vector1=np.array(vector).astype(float)
#vector2=vector1.T
#CTtriad3=CTtriad2.astype(np.float) 
data_PseAAC=pd.DataFrame(data=vector)
data_PseAAC.to_csv('structure_C_ic.csv')
def TAB(hsa_list,pathname):
    jieguo=read_spd_C(hsa_list,pathname)
    vector=jieguo[:,:80]
    return vector
def SPB(hsa_list,pathname):
    jieguo=read_spd_C(hsa_list,pathname)
    vector=jieguo[:,80:]
    return vector
