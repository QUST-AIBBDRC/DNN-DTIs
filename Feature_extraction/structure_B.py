import re
import numpy as np
import pandas as pd

#result = [[0 for x in range(12)] for y in range(2)]

#p=0
#pathname='nr_protein_fasta/'
#zifu='s'+str(p)
#zifu2=pathname+zifu+'.spd33'
#file_read = open(zifu2)
#num_lines = sum(1 for line in open(zifu2))
result=[]
pathname='ic_protein_fasta/'

#def read_spd_file(file):
#    with open(file) as f:
#         records=f.read()
#    record=records.split('\n')[1:]
#    return record
#H=read_spd_file(zifu2)
#t = open('HBPPI_fasta_PB_newname.txt','w')
#for i in range(len(new_sequence)):
#    zifu='>000000'
#    zifuchuan=zifu+str(i)
#    t.write(new_name[i] +'\n')
#    #data_shu=re.sub('N', '', new_sequence[i])
#    data_shu= new_sequence[i]
#    t.write(data_shu+'\n')
#    zifu=[]
#t.close()


def read_spd_B(pathname):

    for i in range(50):
        j=0
        zifu='s'+str(i)
        zifu2=pathname+zifu+'.spd33'
        file_read = open(zifu2)
        num_lines = sum(1 for line in open(zifu2)) 
        i = 0
        a = []
        while i < num_lines:
              i += 1
              str1 = file_read.readline()
              l = str1.rstrip('\n').split()
              H_=l
              H_H=list(H_)
              a.append(H_H)
              H_H=[]
              H_=[]
        a.pop(0)
        num_lines=len(a)
        C_count = 0
        E_count = 0
        H_count = 0
        ASA_sum = 0
        psi_sin_sum = 0
        psi_cos_sum = 0
        phi_sin_sum = 0
        phi_cos_sum = 0
        theta_sin_sum = 0
        theta_cos_sum = 0
        tau_sin_sum = 0
        tau_cos_sum = 0
        length_counter = 0
        for c in range(len(a)):
              length_counter += 1
              file = a[c]
              if file[2] == 'C':
                 C_count += 1
              if file[2] == 'H':
                 H_count += 1
              if file[2] == 'E':
                 E_count += 1
              ASA_sum += float(file[3])
              phi_sin_sum += np.sin(float(file[4]))
              phi_cos_sum += np.cos(float(file[4]))
              psi_sin_sum += np.sin(float(file[5]))
              psi_cos_sum += np.sin(float(file[5]))
              theta_sin_sum += np.sin(float(file[6]))
              theta_cos_sum += np.sin(float(file[6]))
              tau_sin_sum += np.sin(float(file[7]))
              tau_cos_sum += np.cos(float(file[7]))

        C_count = (C_count/length_counter)
        E_count = (E_count/length_counter)
        H_count = (H_count/length_counter)
        ASA_sum = (ASA_sum/length_counter)
        phi_sin_sum = (phi_sin_sum/length_counter)
        phi_cos_sum = (phi_cos_sum / length_counter)
        psi_sin_sum = (psi_sin_sum / length_counter)
        psi_cos_sum = (psi_cos_sum / length_counter)
        theta_sin_sum = (theta_sin_sum/ length_counter)
        theta_cos_sum = (theta_cos_sum / length_counter)
        tau_sin_sum = (tau_sin_sum/length_counter)
        tau_cos_sum = (tau_cos_sum / length_counter)
        datas  = [C_count, E_count, H_count, ASA_sum, phi_sin_sum, phi_cos_sum ,psi_sin_sum , psi_sin_sum ,theta_sin_sum , theta_cos_sum, tau_sin_sum,tau_cos_sum]
        #datas=np.array(datas).astype(float)
        result.append(datas)
        j=j+1
    return  result,j,i,datas
result,j,i,datas=read_spd_B(pathname)
#vector1=np.array(vector).astype(float)
#vector2=vector1.T
#CTtriad3=CTtriad2.astype(np.float) 
data_PseAAC=pd.DataFrame(data=result)
data_PseAAC.to_csv('structure_B_ic.csv')

def SSC(hsa_list,pathname):
    jieguo=read_spd_B(hsa_list,pathname)
    vector=jieguo[:,:3]
    return vector
def ASA(hsa_list,pathname):
    jieguo=read_spd_B(hsa_list,pathname)
    vector=jieguo[:,3]
    return vector
def TAC(hsa_list,pathname):
    jieguo=read_spd_B(hsa_list,pathname)
    vector=jieguo[:,3:]
    return vector

