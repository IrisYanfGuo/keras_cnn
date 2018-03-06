import os
import pandas as pd
import numpy as np



def read_directory(filename,root):
    f = open(filename)
    a = f.read()
    f.close()
    res=[]
    content = a.strip().split()
    for temp in content:
        temp2 = temp.split('/')
        speaker = temp2[1].split('_')[0]
        word = temp2[0]
        dir = root+'/'+temp

        res.append([word,speaker,dir,temp])


    return pd.DataFrame(res, columns=['word', 'speaker', 'path','temp'])



def main():
    direc = read_directory("./file_list.txt", "./trimSound")

    sel_word = direc['word'].unique()
    print(sel_word)

    sel_word = np.append(sel_word[0:14], sel_word[15:])

    for i in sel_word:
        try:
            os.mkdir("./STFTPickle/" + i)
        except:
            pass


if __name__ == '__main__':
    main()