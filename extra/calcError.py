from  __future__ import division
import numpy as np
import difflib
import os
#in_file = 'temp.txt'


def getVal(start_idx, max_val, stop = False):
    inp = []
    out = []
    while (f[start_idx] != "OUTPUT\n"):
        start_idx += 1

    start_idx += 1
    #print 'sid after inp: ', start_idx

    while (f[start_idx] != "EXPECTED\n"):
        val = f[start_idx].split(' ')
        for i in val:
            inp.append(i)
        start_idx += 1

    start_idx += 1

    while (f[start_idx] != "INPUT\n"):
        val = f[start_idx].split(' ')
        for i in val:
            out.append(i)
        start_idx += 1
        if start_idx >= max_val:
            stop = True
            break

    start_idx += 1

    return inp, out, start_idx, stop

def getDifference(str1, str2):
    match_count = 0
    res = difflib.SequenceMatcher(None,str1, str2).get_matching_blocks()
    for block in res:
        match_count += block.size
    diff = max(len(str1), len(str2)) - match_count
    return diff


def getError(inp, out):
    length = 0
    diff = 0
    len_inp = len(inp)
    len_out = len(out)
    if len_inp >= len_out:
        for i in range(len_inp):
            if i < len_out:
                current_out = out[i]
                current_inp = inp[i]
                length = length + len(current_out)
                diff = diff + getDifference(current_out, current_inp)
            #else:
            #    current_inp = inp[i]
            #    diff = diff + len(current_inp)

    else:
        for i in range(len_out):
            if i < len_inp:
                current_out = out[i]
                current_inp = inp[i]
                length = length + len(current_out)
                diff = diff + getDifference(current_out, current_inp)
            else:
                current_out = out[i]
                diff = diff + len(current_out)
                length = length + len(current_out)

    error = diff/length
    return error


in_dir = '/home/nikhil/myCode/git/machine_translation_for_programming_language/test_nesting_length/outputs/'
out_dir = '/home/nikhil/myCode/git/machine_translation_for_programming_language/test_nesting_length/results/'
out_file = out_dir + 'errorNew.txt'


for files in os.listdir(in_dir):
    nesting_loc = files.find('nesting_')
    length_loc = files.find('_length')
    nesting_val = files[nesting_loc+8:length_loc]
    length_val = files[length_loc+8:-4]

    with open(in_dir+files) as f:
        f = f.readlines()

    max_val = len(f)
    start_idx = 0
    error = 0
    num_samples = 0

    while True:
        inp, out, start_idx, stop = getVal(start_idx, max_val)
        num_samples += 1
        #print 'inp: ', inp
        #print 'op: ', out

        #Get error
        error = error + getError(inp, out)
        #print 'er: ', error
        if stop:
            break

    error = error/num_samples

    out_text = "nesting = {} length = {} error = {}\n".format(nesting_val, length_val, error)

    with open(out_file, "a") as f:
        f.write(out_text)

    print "wrote nesting {} length {} for {} samples".format(nesting_val, length_val, num_samples)
