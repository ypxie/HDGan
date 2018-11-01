import glob
import os
import torchfile as torchf
import pickle


path = 'Data/coco/'
save_path = path + 'train/'

t7list = glob.glob(os.path.join(path, 'train2014_ex_t7/*.t7'))
# {'char': not used , 'txt': embedding, 'img': name of file }

embeding_fn = 'char-CNN-RNN-embeddings.pickle' # list([numofembedding, 1024])
file_info = 'filenames.pickle' # list('filename.jpg')

# image need to save in folder not 
print ('find {} files'.format(len(t7list)))
File_info = []
Embed = []
k = 0
for f in t7list:
    data = torchf.load(f)   
    #import pdb; pdb.set_trace()
    name = data[b'img']
    embeddings = data[b'txt']
    Embed.append(embeddings.copy())
    File_info.append(name)
    k += 1
    if k % 100 == 0:
        print('append {}'.format(k),  name)

pickle.dump(File_info, open(os.path.join(save_path, file_info), 'wb'))
pickle.dump(Embed, open(os.path.join(save_path, embeding_fn),  'wb'))

