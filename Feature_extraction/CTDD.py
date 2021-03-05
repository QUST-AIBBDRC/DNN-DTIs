import re, math
import readFasta
import numpy as np
import pandas as pd
import scipy.io as sio

def Count(aaSet, sequence):
    number=0
    code=[]
    select=[]
    for aa in sequence:
        if aa in aaSet:
            number=number+1
    cutoffNums=[1, math.floor(0.25*number),math.floor(0.5*number),math.floor(0.75*number),number]
    myCount=0
    for i in range(len(sequence)):
        if sequence[i] in aaSet:
            myCount=myCount+1
            if myCount in cutoffNums:
                code.append((i+1)/len(sequence))
    if len(code)<5:
        code=[]
        for i in range(len(sequence)):          
            if sequence[i] in aaSet:
               select.append(i)
        if len(select)<1:
            code=[0,0,0,0,0]  
        else:
            if 0 in cutoffNums:
               cutoffNums=np.array(cutoffNums)
               cutoffNums[cutoffNums==0]=1
            for j in range(5):
                label=select[cutoffNums[j]-1]
                code.append((label+1)/len(sequence))
                label=[]
    return code
def CTDD(fastas, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for g in ('1', '2', '3'):
			for d in ['0', '25', '50', '75', '100']:
				header.append(p + '.' + g + '.residue' + d)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('X', '', i[1])
		code = [name]
		for p in property:
			code = code + Count(group1[p], sequence) + Count(group2[p], sequence) + Count(group3[p], sequence)
		encodings.append(code)
	return encodings

fastas = readFasta.readFasta("ic_protein_fasta.txt")
data_CTDD=CTDD(fastas)
data_D1=np.array(data_CTDD)
data_D2=data_D1[1:,1:] 
data_D=data_D2.astype(np.float) 
data_PseAAC=pd.DataFrame(data=data_D)
data_PseAAC.to_csv('CTDD_ic.csv')



