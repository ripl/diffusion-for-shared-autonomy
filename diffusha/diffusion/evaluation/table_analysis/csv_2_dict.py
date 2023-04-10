import string
import numpy as np

def csv_2_dict(file_name: string) -> dict:
    data={}
    with open(file_name) as file:
        for index,line in enumerate(file):
            if "\"" in line:
                continue
            elif 'FWR' in line:
                line = line.split("=")
                fwr = float(line[1][:-1])
                p = float(line[2])
            elif 'actor' in line:
                line = line.split(",")
                if 'Not' in line[-1]:
                    assisted = False
                else:
                    assisted = True
            else:
                line = line.split(",")
                actor = line[0]
                key = str(fwr)+str(p)+actor+str(assisted)
                curr_entry = Entry()
                curr_entry.fwr = fwr
                curr_entry.p = p
                curr_entry.actor = actor
                curr_entry.reward = get_mean_std(line[1])
                curr_entry.avg_reward = get_mean_std(line[2])
                curr_entry.ep_length = get_mean_std(line[3])
                curr_entry.action_diffs = get_mean_std(line[4])
                curr_entry.left = get_num(line[5])
                curr_entry.right = get_num(line[6])
                curr_entry.timeout = get_num(line[7])
                curr_entry.success = get_num(line[5])
                curr_entry.game_over = get_num(line[6])
                curr_entry.floating = get_num(line[7])
                data[key]=curr_entry
    return data

#key:{FWR:0.7,P=0.7,actor='expert',assisted=True}
#value:{reward:xxx,avg_reward:xxx,...}

class Entry:
    def __init__(self) -> None:
        self.fwr = 0
        self.p = 0
        self.actor = 0
        self.reward = [] # list: mean, std
        self.avg_reward = []
        self.ep_length = []
        self.action_diffs = []
        self.left = [] # list: left_num, total_num
        self.right = []
        self.timeout = []
        self.success = []
        self.game_over = []
        self.floating = []

def get_mean_std(s):
    s = s.split("+-")
    if 'n' in s[0]:
        return [float(0), float(0)]
    else:
        return [float(s[0]), float(s[1])]

def get_num(s):
    s = s.split("/")
    if 'n' in s[0]:
        return [float(0), float(0)]
    else:
        return [int(s[0]), int(s[1])]