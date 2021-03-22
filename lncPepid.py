#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter
import math
import numpy as np
import csv
import sys
import joblib
import sklearn.metrics as metrics
from sklearn.preprocessing import minmax_scale

def AAC(fastas, **kw):
	# AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

def GAAC(fastas, **kw):
	group = {
		'alphatic': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharge': 'KRH',
		'negativecharge': 'DE',
		'uncharge': 'STCPNQ'
	}

	groupKey = group.keys()

	encodings = []
	header = ['#']
	for key in groupKey:
		header.append(key)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		count = Counter(sequence)
		myDict = {}
		for key in groupKey:
			for aa in group[key]:
				myDict[key] = myDict.get(key, 0) + count[aa]

		for key in groupKey:
			code.append(myDict[key]/len(sequence))
		encodings.append(code)

	return encodings

def GTPC(fastas, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()
	baseNum = len(groupKey)
	triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	encodings = []
	header = ['#'] + triple
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])

		code = [name]
		myDict = {}
		for t in triple:
			myDict[t] = 0

		sum = 0
		for j in range(len(sequence) - 3 + 1):
			myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] + 1
			sum = sum +1

		if sum == 0:
			for t in triple:
				code.append(0)
		else:
			for t in triple:
				code.append(myDict[t]/sum)
		encodings.append(code)

	return encodings

def DPC(fastas, **kw):
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		code = code + tmpCode
		encodings.append(code)
	return encodings

def CTDC_Count(seq1, seq2):
	sum = 0
	for aa in seq1:
		sum = sum + seq2.count(aa)
	return sum


def CTDC(fastas, **kw):
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
		for g in range(1, len(groups) + 1):
			header.append(p + '.G' + str(g))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		for p in property:
			c1 = CTDC_Count(group1[p], sequence) / len(sequence)
			c2 = CTDC_Count(group2[p], sequence) / len(sequence)
			c3 = 1 - c1 - c2
			code = code + [c1, c2, c3]
		encodings.append(code)
	return encodings

def CTDT(fastas, **kw):
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
		for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
			header.append(p + '.' + tr)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
		for p in property:
			c1221, c1331, c2332 = 0, 0, 0
			for pair in aaPair:
				if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
					c1221 = c1221 + 1
					continue
				if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
					c1331 = c1331 + 1
					continue
				if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
					c2332 = c2332 + 1
			code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
		encodings.append(code)
	return encodings

def CTDD_Count(aaSet, sequence):
	number = 0
	for aa in sequence:
		if aa in aaSet:
			number = number + 1
	cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
	cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

	code = []
	for cutoff in cutoffNums:
		myCount = 0
		for i in range(len(sequence)):
			if sequence[i] in aaSet:
				myCount += 1
				if myCount == cutoff:
					code.append((i + 1) / len(sequence) * 100)
					break
		if myCount == 0:
			code.append(0)
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
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		for p in property:
			code = code + CTDD_Count(group1[p], sequence) + CTDD_Count(group2[p], sequence) + CTDD_Count(group3[p], sequence)
		encodings.append(code)
	return encodings


def construct_kmer():
	ntarr = ('D', 'E', 'K', 'R', 'A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

	kmerArray = []

	for n in range(20):
		str1 = ntarr[n]
		for m in range(20):
			str2 = str1 + ntarr[m]
			kmerArray.append(str2)
	return kmerArray

# single nucleic ggap
def g_gap_single(seq, ggaparray, g):
	# seq length is fix =23

	rst = np.zeros((400))
	for i in range(len(seq) - 1 - g):
		str1 = seq[i]
		str2 = seq[i + 1 + g]
		idx = ggaparray.index(str1 + str2)
		rst[idx] += 1

	for j in range(len(ggaparray)):
		rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

	return rst

def ggap_encode(seq, ggaparray, g):
	result = []
	for x in seq:
		temp = g_gap_single(x[1], ggaparray, g)
		result.append(temp)
	result = np.array(result)
	return result

def GGAP(fastas, **kw):
	kmerArray = construct_kmer()
	encodings = ggap_encode(fastas, kmerArray, 1).tolist()
	kmerArray.insert(0, '#')
	for i in range(0, len(encodings)):
		encodings[i].insert(0, fastas[i][0])
	encodings.insert(0, kmerArray)
	return encodings

def read_file(rf):
	seq = list()
	ids = list()
	
	with open(rf, "r") as lines:
		for data in lines:
			line = data.strip()
			if line[0] == '>':
				ids.append(line[1:])
			else:
				seq.append(line)
	datas = [[x, y] for x, y in zip(ids, seq)]
	
	return datas

def feature_integration(fastas):
	AA = AAC(fastas)
	DP = DPC(fastas)
	GGA = GGAP(fastas)
	GAA = GAAC(fastas)
	GTP = GTPC(fastas)
	CC = CTDC(fastas)
	CT = CTDT(fastas)
	CD = CTDD(fastas)
	CTD = np.column_stack((np.array(CC), np.array(CT)[:,1:], np.array(CD)[:,1:]))
	seq = np.column_stack((np.array(AA), np.array(DP)[:,1:], np.array(GGA)[:,1:]))
	phy = np.column_stack((CTD, np.array(GAA)[:,1:], np.array(GTP)[:,1:]))
	return phy, seq

def feature_selection(fs, features):
    results = list()
    features = np.array(features)
    for i in range(len(features[0])):
        if features[0][i] in fs:
            results.append(features[:, i].tolist())
        else:
            continue
    results = np.array(results).T.tolist()
    return results


if __name__ == '__main__':
	rf = sys.argv[1]
	pos = int(sys.argv[2])
	neg = int(sys.argv[3])
	y_true = np.ones(pos).tolist() + np.zeros(neg).tolist()
	fastas = read_file(rf)
	
	phy, seq = feature_integration(fastas)
	
	fs1 = ['RF','RL.1','YI.1','KI.1','VI','PI.1','LN','IV.1','ML','VA.1','FI.1','MR','AS','LI.1','SY','YT','MF','QA','PL.1','MV.1','MY','SI','PI','HK','LT.1','IN','PP.1','SF','IF','LY.1','KE.1','LY','SY.1','GP','QN','LP','LS','SP','LC','EH','CG','RC.1','LM.1','VD','MD.1','PT.1','DL.1','IT','YL','SG','MI','GS','MH','ST','RY.1','MG.1','GL.1','LG','MH.1','KL','GT','PL','EA','VY','GH','LP.1','TC','MN.1','LR.1','KQ.1','RS','YI','CT','IG.1','QS','NK','VT.1','LM','KV.1','PG.1','SG.1','QA.1','HI','NT','LC.1','AM.1','SN.1','AE','YM','KV','SN','SC.1','NL.1','TP','VP.1','RG','LD.1','TF.1','QW','PN','NF.1','FV','PA.1','TF','RE.1','QT','DL','MV','QF','LW','MM','SR','KS','AR.1','KS.1','LF.1','SP.1','EL','AE.1','RT.1','KF.1','PG','FV.1','SM','TP.1','DT.1','PC','CF.1','LF','QG.1','YC','GA.1','IL','DS.1','VS.1','CL.1','QS.1','LS.1','IR','RL','VV.1','LH','KN','SH','MC','KE','TE','IG','SC',
          'AD','VC','SQ','CV','AP.1','KC','QT.1','TA','NQ.1','EA.1','RR.1','AM','ER.1','HS','FS.1','LE','RA','AK','GT.1','KD','AV.1','GK','RK.1','VS','PW','IY.1','FE','AI','PH','KG','DR.1','NP.1','SV','PH.1','AH','MK.1','ED.1','VC.1','SW','QL','LR','GQ.1','MQ','DE.1','LN.1','KQ','RM.1','NS.1','TH','WD.1','QP.1','FH','QM.1','GN.1','LH.1','PV','VV','FP','AF','HE','SW.1','AS.1','AT.1','GC.1','SR.1','QE','QD','NL','NE','SS','MA','IA.1','SD','IQ','AP','NS','PK.1','VQ.1','AH.1','AL.1','FL.1','TL.1','LG.1','TR','DA.1','KI','RD.1','HA.1','RK','MC.1','RN.1','CA.1','KN.1','LA.1','NM.1','RV.1','QM','TN','DV.1','VA','KG.1','TW.1','HT','MG','NC.1','HR.1','NA','PY','NI','IM','ML.1','SQ.1','QN.1','IP','LV.1','GA','MT','LL','AK.1','VR','MM.1','FY.1','YL.1','ND.1','QG','VN.1','LQ.1','NH','TI.1','QV.1','PN.1','FW','TM.1','FS','AC','RW.1','AW','EI','YS.1','KM.1','IQ.1','DK','FC','HD','GN',
          'LI','SA','QH.1','GL','QH','QI.1','PW.1','QR','KW.1','ME.1','TT.1','ME','VT','AF.1','AQ','KK','TM','RQ.1','KH.1','IH.1','AG.1','TQ','CM.1','PE.1','ST.1','FG','KM','QV','NT.1','YV','VW.1','HS.1','FD','WL.1','AW.1','GM','GW.1','VH.1','TI','YS','HP.1','MP.1','GI','GH.1','YK','FW.1','KC.1','HL','VQ','MK','IN.1','WI.1','HF.1','YM.1','SE','SD.1','CS.1','MQ.1','RM','SF.1','AV','GE.1','DK.1','MF.1','SK.1','SL','DP.1','PP','PF','MT.1','II','VD.1','SK','FF.1','VI.1','KT.1','EE.1','TA.1','FM.1','KA','VL.1','FT','GP.1','CC','LQ','LA','NF','ES','DM.1','GV','AL','GI.1','TG','SA.1','KL.1','IH','VY.1','KA.1','FN.1','IE','VG','TV','HM','GG','YY','PM','NP','HH.1','IS','KR.1','MN','QQ.1','RR','CS','QD.1','SV.1','RS.1','CK','HK.1','CR.1','NV','DI','YR','CI','PT','NH.1','DV','FL','CG.1','CN','IC.1','NG','HC.1','SL.1','HI.1','VK','PS.1','DY.1','VL','GS.1','ET','CH.1','PQ','PD','WC.1',
          'DR','AA.1','TL','GR.1','HC','HQ.1','II.1','RV','PS','QE.1','EV','AR','EC.1','KR','MD','RA.1','VE.1','VE','SH.1','MW','CV.1','PA','MP','KK.1','FP.1','VF.1','FG.1','IV','NY','YF','FQ','HG.1','WT','TE.1','RI.1','SS.1','HN','PM.1','RC','IM.1','RY','DS','NV.1','WS','LV','ID','IR.1','DI.1','EL.1','QC.1','TR.1','CC.1','CT.1','GR','RE','EW','FC.1','DE','CL','EV.1','ND','EG','EK','PE','CA','FY','KY.1','HP','DD.1','EP','YH.1','CY.1','AD.1','YE.1','YN','TY.1','NR.1','SM.1','YD','KF','WW','VM.1','WY','FF','WA','YV.1','IK','AY.1','RT','EM.1','IF.1','PC.1','W','FD.1','RG.1','YR.1','AQ.1','HH','FM','TC.1','GD.1','WQ','KH','CQ','NK.1','GE','CR','GG.1','CW.1','CQ.1','YQ','LT','QW.1','YF.1','MW.1','TT','YA','GV.1','CY','TK','RF.1','GC','RP','ID.1','IA','DF','NI.1','WQ.1','RH.1','TV.1','LL.1','CP.1','TD','GW','DN','NY.1','TY','QQ','LK','PR.1','DH.1','QC','YC.1','DG','KW','YP.1',
          'KD.1','NR','DF.1','FK','EF.1','EQ','PF.1','VF','D','IT.1','IW','PV.1','YK.1','GY.1','YE','PD.1','IC','EG.1','HT.1','TW','VP','DC','RQ','AN','LD','LW.1','CD','RD','TD.1','QF.1','ED','WR','IW.1','LE.1','EC','PK','MI.1','HQ','A','GK.1','AN.1','DA','GM.1','CP','IP.1','QI','NM','YH','CK.1','RP.1','CN.1','EE','S','IL.1','FN','FQ.1','YG.1','EN','CI.1','TK.1','AY','WT.1','CW','YN.1','HA','PR','QR.1','KT','HY','EQ.1','QY','WP.1','RN','CD.1','WV.1','MS.1','PQ.1','EP.1','AC.1','NW.1','HV.1','DH','WM','GF','WR.1','WH.1','GQ','HR','HL.1','WK','HG','FH.1','DQ.1','TQ.1','IK.1','EF','DN.1','VW','HW','YG','NW','HN.1','YW','TN.1','WE.1','DD','WL','VH','DY','CH','EW.1','YT.1','QY.1','WY.1','TG.1','WI','HW.1','NA.1','WF.1','DW.1','DM','DW','WE','WN.1','SI.1','EY.1','MA.1','KP.1','WF','GD','WA.1','EY','NN.1','HV','MR.1','NN','ES.1','YY.1','YW.1','VM','NG.1','SE.1','QK','ER','CE.1',
          'WK.1','HD.1','EK.1','WM.1','MY.1','IY','QK.1','FA.1','FA','WV','WD','WS.1','E','FI','EI.1','IE.1','VG.1','DT','TH.1','IS.1','GY','KY','AT','QL.1','EN.1','FT.1','CF','NE.1','DQ','CE','FK.1','EH.1','EM','YP','VN','LK.1','Y','HM.1','WG.1','RI','CM','RH','WP','HE.1','ET.1','FR','GF.1','PY.1','HY.1','VK.1','RW','YQ.1','VR.1','YD.1','FE.1','YA.1','TS.1','NC','DC.1','WC','WN','WH','NQ','MS','TS','H','I','AI.1','FR.1','F','V','N','Q','AA','HF','DP','T','WG','L','DG.1','WW.1','C','K','R','M','P','G']

	seqdatas = feature_selection(fs1, seq)
	
	fs2 = ['normwaalsvolume.Tr2332','solventaccess.2.residue50','solventaccess.1.residue100','secondarystruct.3.residue100','hydrophobicity_PONP930101.1.residue25','hydrophobicity_ENGD860101.1.residue50','hydrophobicity_CASG920101.G3','normwaalsvolume.2.residue50','secondarystruct.3.residue25','polarity.3.residue75','hydrophobicity_CASG920101.1.residue50','hydrophobicity_ZIMJ680101.2.residue75','hydrophobicity_ZIMJ680101.2.residue100','polarity.3.residue100','normwaalsvolume.2.residue25','hydrophobicity_ZIMJ680101.2.residue0','hydrophobicity_FASG890101.Tr2332','hydrophobicity_ARGP820101.1.residue75','polarizability.2.residue100','secondarystruct.G3','hydrophobicity_PRAM900101.3.residue100','hydrophobicity_PONP930101.Tr1331','hydrophobicity_PRAM900101.Tr2332','solventaccess.2.residue0','aromatic.postivecharger.uncharger','hydrophobicity_CASG920101.3.residue50','hydrophobicity_FASG890101.1.residue0',
           'polarizability.3.residue25','hydrophobicity_PONP930101.1.residue0','hydrophobicity_CASG920101.2.residue25','hydrophobicity_ZIMJ680101.3.residue25','hydrophobicity_ARGP820101.1.residue25','hydrophobicity_FASG890101.3.residue75','polarity.G3','uncharger.postivecharger.negativecharger','solventaccess.1.residue0','solventaccess.3.residue100','hydrophobicity_PRAM900101.2.residue0','charge.2.residue0','polarity.3.residue25','polarizability.2.residue75','solventaccess.1.residue25','postivecharger.postivecharger.negativecharger','alphaticr.negativecharger.aromatic','hydrophobicity_ENGD860101.1.residue0','polarizability.2.residue0','polarizability.3.residue0','alphaticr.alphaticr.uncharger','hydrophobicity_FASG890101.2.residue100','polarity.Tr1221','polarizability.G1','charge.1.residue50','alphaticr.negativecharger.alphaticr','alphaticr.postivecharger.uncharger','hydrophobicity_PONP930101.1.residue75',
           'hydrophobicity_PONP930101.G2','negativecharger.uncharger.aromatic','secondarystruct.3.residue0','alphaticr.alphaticr.alphaticr','hydrophobicity_ARGP820101.1.residue0','solventaccess.1.residue50','polarizability.Tr2332','hydrophobicity_PRAM900101.2.residue100','hydrophobicity_PRAM900101.3.residue75','hydrophobicity_ARGP820101.G3','normwaalsvolume.1.residue100','uncharger.alphaticr.alphaticr','hydrophobicity_ARGP820101.1.residue50','negativecharger.postivecharger.alphaticr','hydrophobicity_FASG890101.1.residue25','hydrophobicity_PONP930101.2.residue100','charge.Tr1331','charge.G1','alphaticr.negativecharger.uncharger','hydrophobicity_PONP930101.2.residue25','normwaalsvolume.1.residue50','postivecharger.aromatic.uncharger','alphaticr.alphaticr.postivecharger','hydrophobicity_ENGD860101.Tr1221','normwaalsvolume.Tr1331','charge.3.residue100','postivecharger.postivecharger.uncharger','aromatic.negativecharger.uncharger',
           'hydrophobicity_ZIMJ680101.G3','secondarystruct.3.residue50','hydrophobicity_ENGD860101.1.residue100','secondarystruct.Tr1221','uncharger.aromatic.postivecharger','uncharger.postivecharger.postivecharger','hydrophobicity_CASG920101.2.residue100','charge.Tr1221','hydrophobicity_ENGD860101.1.residue75','polarizability.2.residue50','negativecharger.postivecharger.negativecharger','normwaalsvolume.1.residue0','uncharger.uncharger.alphaticr','postivecharger.negativecharger.alphaticr','charge.2.residue25','negativecharger.aromatic.uncharger','postivecharger.negativecharger.uncharger','uncharger.alphaticr.postivecharger','normwaalsvolume.G2','solventaccess.3.residue50','polarizability.1.residue25','aromatic.negativecharger.alphaticr','hydrophobicity_FASG890101.Tr1331','hydrophobicity_ENGD860101.3.residue50','solventaccess.3.residue25','hydrophobicity_CASG920101.3.residue75','hydrophobicity_ARGP820101.2.residue75','solventaccess.3.residue75',
           'hydrophobicity_CASG920101.2.residue75','uncharger.postivecharger.uncharger','polarity.Tr1331','secondarystruct.2.residue100','hydrophobicity_CASG920101.1.residue75','hydrophobicity_ENGD860101.G3','hydrophobicity_PONP930101.Tr1221','hydrophobicity_PRAM900101.Tr1331','hydrophobicity_FASG890101.2.residue75','solventaccess.3.residue0','postivecharger.alphaticr.uncharger','uncharger.postivecharger.alphaticr','hydrophobicity_ENGD860101.Tr1331','hydrophobicity_PONP930101.1.residue100','postivecharger.aromatic.aromatic','solventaccess.G3','aromatic.negativecharger.aromatic','secondarystruct.2.residue50','postivecharger.uncharger.alphaticr','normwaalsvolume.2.residue75','hydrophobicity_CASG920101.Tr2332','hydrophobicity_ARGP820101.Tr1221','hydrophobicity_ZIMJ680101.1.residue50','uncharger.negativecharger.negativecharger','negativecharger.postivecharger.aromatic','uncharger.negativecharger.alphaticr','secondarystruct.G1','normwaalsvolume.1.residue25',
           'normwaalsvolume.2.residue0','alphaticr.postivecharger.negativecharger','hydrophobicity_ENGD860101.2.residue100','hydrophobicity_PRAM900101.2.residue25','hydrophobicity_PONP930101.1.residue50','hydrophobicity_CASG920101.G2','hydrophobicity_ZIMJ680101.Tr2332','polarizability.G2','polarizability.2.residue25','hydrophobicity_CASG920101.Tr1331','hydrophobicity_ARGP820101.1.residue100','uncharger.aromatic.aromatic','hydrophobicity_ZIMJ680101.G2','alphaticr.negativecharger.postivecharger','postivecharger.aromatic.postivecharger','postivecharger.uncharger.postivecharger','aromatic.aromatic.uncharger','postivecharger.postivecharger.alphaticr','charge.G2','hydrophobicity_ARGP820101.G1','aromatic.negativecharger.negativecharger','alphaticr.aromatic.postivecharger','postivecharger.alphaticr.postivecharger','hydrophobicity_ENGD860101.Tr2332','hydrophobicity_FASG890101.1.residue50','hydrophobicity_ARGP820101.2.residue25','polarizability.1.residue0',
           'postivecharger.aromatic.negativecharger','polarizability.Tr1221','secondarystruct.2.residue75','hydrophobicity_CASG920101.3.residue25','secondarystruct.2.residue0','secondarystruct.G2','hydrophobicity_ZIMJ680101.2.residue50','postivecharger.alphaticr.aromatic','polarity.2.residue50','hydrophobicity_PRAM900101.2.residue50','solventaccess.Tr1221','hydrophobicity_CASG920101.2.residue50','hydrophobicity_ENGD860101.3.residue25','hydrophobicity_ARGP820101.G2','secondarystruct.Tr2332','hydrophobicity_ZIMJ680101.Tr1331','hydrophobicity_ARGP820101.Tr2332','hydrophobicity_ENGD860101.2.residue50','secondarystruct.1.residue0','hydrophobicity_ARGP820101.2.residue50','polarizability.1.residue100','hydrophobicity_PRAM900101.Tr1221','hydrophobicity_FASG890101.G2','alphatic','hydrophobicity_FASG890101.G1','secondarystruct.1.residue50','alphaticr.uncharger.alphaticr','hydrophobicity_ENGD860101.G1','normwaalsvolume.2.residue100','uncharger.aromatic.alphaticr',
           'alphaticr.uncharger.postivecharger','uncharger.uncharger.postivecharger','hydrophobicity_ARGP820101.3.residue25','aromatic.alphaticr.alphaticr','hydrophobicity_ZIMJ680101.2.residue25','hydrophobicity_ARGP820101.3.residue75','hydrophobicity_CASG920101.3.residue100','alphaticr.negativecharger.negativecharger','hydrophobicity_FASG890101.1.residue100','hydrophobicity_ZIMJ680101.1.residue75','polarity.Tr2332','hydrophobicity_ZIMJ680101.1.residue25','hydrophobicity_ARGP820101.3.residue50','hydrophobicity_PONP930101.2.residue0','charge.Tr2332','hydrophobicity_FASG890101.1.residue75','hydrophobicity_FASG890101.2.residue50','charge.1.residue75','aromatic.uncharger.aromatic','aromatic.uncharger.alphaticr','solventaccess.1.residue75','hydrophobicity_FASG890101.Tr1221','negativecharger.alphaticr.alphaticr','hydrophobicity_CASG920101.Tr1221','uncharger.uncharger.aromatic','uncharger.alphaticr.aromatic','postivecharger.uncharger.aromatic','hydrophobicity_ARGP820101.3.residue100',
           'hydrophobicity_FASG890101.3.residue25','postivecharger.alphaticr.alphaticr','hydrophobicity_FASG890101.2.residue25','polarizability.Tr1331','alphaticr.postivecharger.alphaticr','hydrophobicity_ZIMJ680101.3.residue75','hydrophobicity_ARGP820101.Tr1331','solventaccess.2.residue25','alphaticr.aromatic.negativecharger','solventaccess.Tr1331','hydrophobicity_PONP930101.G1','uncharger.alphaticr.uncharger','secondarystruct.3.residue75','alphaticr.aromatic.uncharger','polarity.2.residue25','polarity.G2','hydrophobicity_FASG890101.3.residue50','solventaccess.G1','uncharger.uncharger.negativecharger','hydrophobicity_ENGD860101.G2','charge.2.residue50','postivecharger.negativecharger.negativecharger','postivecharge','hydrophobicity_ZIMJ680101.G1','aromatic.uncharger.negativecharger','aromatic.postivecharger.postivecharger','aromatic','negativecharger.negativecharger.alphaticr','hydrophobicity_FASG890101.3.residue100','postivecharger.uncharger.negativecharger',
           'alphaticr.postivecharger.postivecharger','negativecharger.uncharger.postivecharger','uncharger.aromatic.uncharger','hydrophobicity_ENGD860101.2.residue75','alphaticr.aromatic.alphaticr','postivecharger.postivecharger.postivecharger','polarity.3.residue0','hydrophobicity_ZIMJ680101.Tr1221','aromatic.postivecharger.alphaticr','hydrophobicity_ARGP820101.3.residue0','negativecharger.alphaticr.aromatic','hydrophobicity_PONP930101.Tr2332','aromatic.alphaticr.uncharger','postivecharger.alphaticr.negativecharger','polarity.2.residue100','hydrophobicity_CASG920101.3.residue0','polarity.1.residue100','polarizability.1.residue50','polarity.1.residue25','polarizability.3.residue50','uncharger.aromatic.negativecharger','hydrophobicity_ZIMJ680101.1.residue0','hydrophobicity_ZIMJ680101.3.residue0','hydrophobicity_PRAM900101.3.residue25','hydrophobicity_ARGP820101.2.residue100','hydrophobicity_ZIMJ680101.3.residue100','normwaalsvolume.G1','uncharger.negativecharger.postivecharger',
           'aromatic.alphaticr.postivecharger','negativecharger.alphaticr.uncharger','negativecharger.uncharger.alphaticr','aromatic.postivecharger.aromatic','hydrophobicity_FASG890101.2.residue0','hydrophobicity_ENGD860101.2.residue0','negativecharger.postivecharger.uncharger','hydrophobicity_ENGD860101.2.residue25','negativecharger.postivecharger.postivecharger','secondarystruct.Tr1331','hydrophobicity_ENGD860101.3.residue100','secondarystruct.1.residue75','solventaccess.G2','charge.3.residue75','hydrophobicity_ENGD860101.3.residue75','alphaticr.uncharger.uncharger','hydrophobicity_PRAM900101.G2','uncharger.negativecharger.uncharger','hydrophobicity_CASG920101.1.residue25','aromatic.uncharger.uncharger','polarizability.3.residue100','negativecharger.alphaticr.negativecharger','uncharge','aromatic.alphaticr.aromatic','negativecharger.aromatic.alphaticr','hydrophobicity_ENGD860101.3.residue0','normwaalsvolume.Tr1221','uncharger.uncharger.uncharger','hydrophobicity_FASG890101.3.residue0',
           'hydrophobicity_PONP930101.2.residue50','uncharger.negativecharger.aromatic','aromatic.postivecharger.negativecharger','hydrophobicity_ZIMJ680101.3.residue50','charge.2.residue75','alphaticr.aromatic.aromatic','secondarystruct.2.residue25','charge.1.residue25','aromatic.uncharger.postivecharger','postivecharger.postivecharger.aromatic','hydrophobicity_PRAM900101.3.residue50','aromatic.aromatic.negativecharger','uncharger.postivecharger.aromatic','postivecharger.uncharger.uncharger','solventaccess.2.residue75','uncharger.alphaticr.negativecharger','hydrophobicity_CASG920101.2.residue0','negativecharger.alphaticr.postivecharger','charge.1.residue100','hydrophobicity_PONP930101.2.residue75','alphaticr.alphaticr.negativecharger','polarizability.3.residue75','negativecharger.uncharger.uncharger','polarity.3.residue50','secondarystruct.1.residue25','negativecharger.aromatic.postivecharger','hydrophobicity_CASG920101.G1','secondarystruct.1.residue100','hydrophobicity_PRAM900101.2.residue75',
           'postivecharger.negativecharger.aromatic','polarity.2.residue75','hydrophobicity_ENGD860101.1.residue25','negativecharger.aromatic.negativecharger','charge.G3','postivecharger.negativecharger.postivecharger','alphaticr.postivecharger.aromatic','hydrophobicity_ZIMJ680101.1.residue100','solventaccess.2.residue100','polarity.1.residue50','negativecharger.negativecharger.uncharger','alphaticr.alphaticr.aromatic','polarity.2.residue0','charge.3.residue50','charge.1.residue0','polarity.1.residue75','hydrophobicity_PRAM900101.G3','aromatic.negativecharger.postivecharger','postivecharger.aromatic.alphaticr','alphaticr.uncharger.negativecharger','negativecharger.negativecharger.negativecharger','aromatic.alphaticr.negativecharger','aromatic.aromatic.postivecharger','polarizability.G3','hydrophobicity_PONP930101.G3','normwaalsvolume.1.residue75','solventaccess.Tr2332','hydrophobicity_ARGP820101.2.residue0','charge.2.residue100','hydrophobicity_CASG920101.1.residue100',
           'negativecharger.negativecharger.postivecharger','polarizability.1.residue75','charge.3.residue25','aromatic.aromatic.alphaticr','hydrophobicity_FASG890101.G3','alphaticr.uncharger.aromatic','polarity.1.residue0','negativecharger.uncharger.negativecharger','aromatic.aromatic.aromatic','negativecharger.aromatic.aromatic','hydrophobicity_CASG920101.1.residue0','normwaalsvolume.3.residue100','charge.3.residue0','polarity.G1','hydrophobicity_PRAM900101.1.residue100','normwaalsvolume.3.residue0']

	phydatas = feature_selection(fs2, phy)

	test_x = np.column_stack((np.array(seq)[1:,1:], np.array(phy)[1:,1:]))
	test_x = minmax_scale((test_x))
	model = joblib.load("evmmodel.m")
	y_pred = model.predict(test_x)
	print(y_pred.tolist())
