from head import *

def read_pickle_dir(filename,root):
    f = open(filename)
    a = f.read()
    f.close()
    res = []
    content = a.strip().split()
    for temp in content:
        temp2 = temp.split('/')
        speaker = temp2[1].split('_')[0]
        word = temp2[0]
        dir = root + '/' + temp+'.pkl'

        res.append([word, speaker, dir, temp])

    return pd.DataFrame(res, columns=['word', 'speaker', 'path', 'temp'])

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

