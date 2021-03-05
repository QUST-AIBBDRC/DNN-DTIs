import numpy
import numpy as np
import pandas as pd


#hsa_list=open("Enzyme_name7.txt")

pathname='ic_protein_fasta/'

result=[]
def read_spd_D(pathname):
    for p in range(50):
        zifu='s'+str(p)
        zifu2=pathname+zifu+'.spd33'
        file_read = open(zifu2)
        num_lines = sum(1 for line in open(zifu2))
        matrix = [[0 for x in range(8)] for y in range(8)]
        probability_matrix = [[0 for x in range(3)] for y in range(3)]
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
                degree_matrix[x][degree_index+1] = numpy.math.cos(float(array[x][y]) * numpy.pi / 180  )
                degree_index += 2
        temp_array = array
        array = degree_matrix
        num_lines -= 1
        for k in range(0, 8):
            for l in range(0, 8):
                for i in range(0, num_lines - 1 ):
                    matrix[k][l] += float(array[i][k]) * float(array[i + 1][l])
                matrix[k][l] = matrix[k][l] / (num_lines - 1)
        array = temp_array
        for k in range(0, 3):
            for l in range(5, 8):
                for i in range(0, num_lines - 1 ):
                    probability_matrix[k][l- 5] += numpy.float(array[i][k]) * numpy.float(array[i + 1][l -5 ])
                probability_matrix[k][l - 5] = probability_matrix[k][l - 5] / (num_lines - 1)
        final_matrix = [[0 for x in range(73)] for y in range(1)]
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
result=read_spd_D(pathname)
result=np.array(result)
vector=np.reshape(result,(result.shape[0],result.shape[1]*result.shape[2]))
#vector1=np.array(vector).astype(float)
#vector2=vector1.T
#CTtriad3=CTtriad2.astype(np.float) 
data_PseAAC=pd.DataFrame(data=vector)
data_PseAAC.to_csv('structure_D_ic.csv')

def TAAC(hsa_list,pathname):
    jieguo=read_spd_D(hsa_list,pathname)
    vector=jieguo[:,:64]
    return vector
def SPAC(hsa_list,pathname):
    jieguo=read_spd_D(hsa_list,pathname)
    vector=jieguo[:,64:]
    return vector
