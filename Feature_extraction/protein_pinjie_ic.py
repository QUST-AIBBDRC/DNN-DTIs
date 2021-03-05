import numpy as np
import pandas as pd

file_structure='out_ic.csv'
structure=pd.read_csv(file_structure,index_col=0)

file1='CTDC_ic.csv'
file2='CTDT_ic.csv'
file3='CTDD_ic.csv'
file4='CT_ic.csv'
file5='PseAAC_ic.csv'
file6='PsePSSM_ic_network.csv'
file7='NMBroto_ic.csv'
file8='structure_B_ic.csv'
file9='structure_C_ic.csv'
file10='structure_D_ic.csv'

csv1=pd.read_csv(file1,index_col=0)
csv2=pd.read_csv(file2,index_col=0)
csv3=pd.read_csv(file3,index_col=0)
csv4=pd.read_csv(file4,index_col=0)
csv5=pd.read_csv(file5,index_col=0)
csv6=pd.read_csv(file6,index_col=0)
csv7=pd.read_csv(file7,index_col=0)
csv8=pd.read_csv(file8,index_col=0)
csv9=pd.read_csv(file9,index_col=0)
csv10=pd.read_csv(file10,index_col=0)

result=pd.concat((csv1,csv2,csv3,csv4,csv5,csv6,csv7,csv8,csv9,csv10),axis=1)
vector_end=np.concatenate((structure,result),axis=1)


csv_data=pd.DataFrame(data=vector_end)
csv_data.to_csv('ic_vector.csv')