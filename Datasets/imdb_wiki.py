import scipy.io as sio
import os
import cPickle as pickle
import numpy as np
import cv2

def read_mat(mat_path,cache_dir='/tmp'):
    # Reading the .mat annotation files in IMDB-WIKI datasets and converting them to python objects.
    # Storing in the cache file (default in `/tmp/imdb_wiki.pth`).
    cache_file = os.path.join(cache_dir,'imdb_wiki.pth')
    if not os.path.isfile(cache_file):
        print "generating cache_file"
        file = sio.loadmat(mat_path)
        image_paths = file['imdb'][0][0]['full_path'][0]
        image_paths = [full_path[0] for full_path in image_paths]
        face_locations = file['imdb'][0][0]['face_location'][0]
        genders = file['imdb'][0][0]['gender'][0]
        genders = [int(gender) if -1<gender<3 else -1 for gender in genders]
        dob = file['imdb'][0][0]['dob'][0]
        photo_taken = file['imdb'][0][0]['photo_taken'][0]
        ages = np.array(photo_taken - dob / 365 + 1,np.uint8)
        face_scores = file['imdb'][0][0]['face_score'][0]
        second_face_scores = file['imdb'][0][0]['second_face_score'][0]
        pickle.dump([image_paths, face_locations, genders, ages,
                     face_scores, second_face_scores], open(cache_file, 'wb'))
    else:
        print "read from cache_file"
        image_paths, face_locations, genders, ages, face_scores, second_face_scores = pickle.load(open(cache_file, 'rb'))
    print "read mat OK"
    return image_paths, face_locations, genders, ages, face_scores, second_face_scores

def crop_image(mat_path,input_dir,output_dir,expand_rate=0,max_size=600):
    image_paths, face_locations, genders, ages, face_score, second_face_score=read_mat(mat_path)
    for image_path,loc in zip(image_paths,face_locations):
        in_path=os.path.join(input_dir,image_path)
        out_path=os.path.join(output_dir,image_path)
        out_dir=os.path.split(out_path)[0]
        if not os.path.exists(out_dir):
            print ("make direction %s"%out_dir)
            os.makedirs(out_dir)
        im=cv2.imread(in_path)
        loc=loc[0].astype(np.int32)
        h = loc[3] - loc[1]
        w = loc[2] - loc[0]
        if expand_rate>0:
            loc[1]-=h*expand_rate
            loc[3]+=h*expand_rate
            loc[0]-=w*expand_rate
            loc[2]+=w*expand_rate
        loc=np.maximum(0,loc)
        loc[3]=np.minimum(im.shape[0],loc[3])
        loc[2]=np.minimum(im.shape[1],loc[2])
        # loc=loc.astype(np.int32)
        im=im[loc[1]:loc[3],loc[0]:loc[2]]
        h = loc[3] - loc[1]
        w = loc[2] - loc[0]
        if w>max_size or h>max_size:
            if w!=h:
                pass
            print("resize picture %s"%image_path)
            resize_factor=np.minimum(1.*max_size/w,1.*max_size/h)
            im=cv2.resize(im,(int(w*resize_factor),int(h*resize_factor)))
        cv2.imwrite(out_path,im)



def generate_caffe_txt_age(mat_path,output_path,age_range,
                           cache_dir='/tmp',ignore_second_face=False,test_ratio=0.2):
    # read mat file then generate train.txt and test.txt for age estimation training.
    # `age_range` is a list containing age range like (0,2) and (32,40).
    image_paths, face_locations, genders, ages, face_scores, second_face_scores = read_mat(mat_path,cache_dir)
    # generate classes
    ages_mid=np.array([(age[0]+age[1])/2 for age in age_range])
    classes=[]
    for age in ages:
        classes.append(np.argmin(np.abs(ages_mid-age)))
        for idx,r in enumerate(age_range):
            if r[0]<=age<=r[1]:
                classes[-1]=idx
                break
    shuffle_idx=np.arange(len(image_paths))
    np.random.shuffle(shuffle_idx)
    train_idx=shuffle_idx[int(len(shuffle_idx)*test_ratio):]
    test_idx=shuffle_idx[:int(len(shuffle_idx)*test_ratio)]
    with open(os.path.join(output_path, 'age_train.txt'), 'w') as trainf:
        for idx in train_idx:
            if ignore_second_face:
                if second_face_scores[idx]>1.5:
                    continue
            trainf.write("%s %d\n" % (image_paths[idx], classes[idx]))
    with open(os.path.join(output_path, 'age_test.txt'), 'w') as testf:
        for idx in test_idx:
            if ignore_second_face:
                if second_face_scores[idx]>1.5:
                    continue
            testf.write("%s %d\n"%(image_paths[idx],classes[idx]))

def generate_caffe_txt_gender(mat_path,output_path,cache_dir='/tmp',test_ratio=0.2):
    # read mat file then generate train.txt and test.txt for gender estimation training
    image_paths, face_locations, genders, ages, face_score, second_face_score = read_mat(mat_path, cache_dir)
    shuffle_idx=np.arange(len(image_paths))
    np.random.shuffle(shuffle_idx)
    train_idx=shuffle_idx[int(len(shuffle_idx)*test_ratio):]
    test_idx=shuffle_idx[:int(len(shuffle_idx)*test_ratio)]
    with open(os.path.join(output_path, 'gender_train.txt'), 'w') as trainf:
        for idx in train_idx:
            if genders[idx]==-1:continue
            trainf.write("%s %d\n" % (image_paths[idx], genders[idx]))
    with open(os.path.join(output_path, 'gender_test.txt'), 'w') as testf:
        for idx in test_idx:
            if genders[idx]==-1:continue
            testf.write("%s %d\n"%(image_paths[idx],genders[idx]))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('mat_path',help='.mat file path in IMDB-WIKI datasets',type=str)
    args=parser.parse_args()
    read_mat(args.mat_path)
