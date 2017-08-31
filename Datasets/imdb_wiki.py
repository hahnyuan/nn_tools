import scipy.io as sio
import os
import cPickle as pickle
import numpy as np

def read_mat(mat_path,cache_dir='/tmp'):
    # Reading the .mat annotation files in IMDB-WIKI datasets and converting them to python objects.
    # Storing in the cache file (default in `/tmp/imdb_wiki.pth`).
    cache_file = os.path.join(cache_dir,'imdb_wiki.pth')
    if not os.path.isfile(cache_file):
        print "generating cache_file"
        file = sio.loadmat(mat_path)
        full_paths = file['imdb'][0][0]['full_path'][0]
        full_paths = [full_path[0] for full_path in full_paths]
        face_locations = file['imdb'][0][0]['face_location'][0]
        genders = file['imdb'][0][0]['gender'][0]
        genders = [int(gender) if -1<gender<3 else -1 for gender in genders]
        dob = file['imdb'][0][0]['dob'][0]
        photo_taken = file['imdb'][0][0]['photo_taken'][0]
        ages = np.array(photo_taken - dob / 365 + 1,np.uint8)
        pickle.dump([full_paths, face_locations, genders, ages], open(cache_file, 'wb'))
    else:
        print "read from cache_file"
        full_paths, face_locations, genders, ages = pickle.load(open(cache_file, 'rb'))
    print "read mat OK"
    return full_paths, face_locations, genders, ages

def generate_caffe_txt_age(mat_path,output_path,cache_dir='/tmp',test_ratio=0.2,age_interval=10):
    # read mat file then generate train.txt and test.txt for age estimation training.
    # each `age_interval` will be a class
    full_paths, face_locations, genders, ages = read_mat(mat_path,cache_dir)
    # generate classes
    classes = [age/age_interval for age in ages]
    shuffle_idx=np.arange(len(full_paths))
    np.random.shuffle(shuffle_idx)
    train_idx=shuffle_idx[int(len(shuffle_idx)*test_ratio):]
    test_idx=shuffle_idx[:int(len(shuffle_idx)*test_ratio)]
    with open(os.path.join(output_path, 'age_train.txt'), 'w') as trainf:
        for idx in train_idx:
            trainf.write("%s %d\n" % (full_paths[idx], classes[idx]))
    with open(os.path.join(output_path, 'age_test.txt'), 'w') as testf:
        for idx in test_idx:
            testf.write("%s %d\n"%(full_paths[idx],classes[idx]))

def generate_caffe_txt_gender(mat_path,output_path,cache_dir='/tmp',test_ratio=0.2):
    # read mat file then generate train.txt and test.txt for gender estimation training
    full_paths, face_locations, genders, ages = read_mat(mat_path, cache_dir)
    shuffle_idx=np.arange(len(full_paths))
    np.random.shuffle(shuffle_idx)
    train_idx=shuffle_idx[int(len(shuffle_idx)*test_ratio):]
    test_idx=shuffle_idx[:int(len(shuffle_idx)*test_ratio)]
    with open(os.path.join(output_path, 'gender_train.txt'), 'w') as trainf:
        for idx in train_idx:
            if genders[idx]==-1:continue
            trainf.write("%s %d\n" % (full_paths[idx], genders[idx]))
    with open(os.path.join(output_path, 'gender_test.txt'), 'w') as testf:
        for idx in test_idx:
            if genders[idx]==-1:continue
            testf.write("%s %d\n"%(full_paths[idx],genders[idx]))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('mat_path',help='.mat file path in IMDB-WIKI datasets',type=str)
    args=parser.parse_args()
    read_mat(args.mat_path)
