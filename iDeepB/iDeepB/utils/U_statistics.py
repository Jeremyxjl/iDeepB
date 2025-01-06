from collections import Counter
import numpy as np

def base_percent_cal(seqList):
    base_count = {}
    base_count["A"]=0
    base_count["C"]=0
    base_count["G"]=0
    base_count["T"]=0
    #base_count["U"]=0
    
    for peak in seqList:
        peak = peak.replace('U', 'T')
        peak_base = Counter(peak)
        base_count["A"] = base_count["A"] + peak_base["A"]
        base_count["C"] = base_count["C"] + peak_base["C"]
        base_count["G"] = base_count["G"] + peak_base["G"]
        base_count["T"] = base_count["T"] + peak_base["T"]
        #base_count["U"] = base_count["U"] + peak_base["T"]
        base_sum = sum(base_count.values())
        base_percent = np.array([v for v in base_count.values()])/base_sum
        base_percent_dict= dict(zip(["A", "C", "G", "T"], base_percent))
    return base_count, base_percent_dict

def U_percent_seqlist(seqlist):
    base_count = base_percent(seqlist)
    base_sum = sum(base_count.values())
    return np.array([v for v in base_count.values()])/base_sum