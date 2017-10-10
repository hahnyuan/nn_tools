import scipy.io as sio
import os
import cPickle as pickle
import numpy as np

AGES_START=[0,4,8,15,25,38,48,60]  # the start of the age scope
AGES_END=[2,6,13,20,32,43,53,100]  # the end of the age scope
AGES_MID=np.array([(i+j)/2 for i,j in zip(AGES_END,AGES_START)])
GENDER=['m','f']  # u for unknown

def get_class(age):
    for i,(s,e) in enumerate(zip(AGES_START,AGES_END)):
        if s<=age<=e:
            return i
    return np.argmin(np.abs(AGES_MID-age))

def read_txt(root_path,cache_dir='/tmp'):
    # Reading the .txt annotation files in adience datasets and converting them to python objects.
    # Storing in the cache file (default in `/tmp/adience.pth`).
    txt_list=['fold_0_data.txt','fold_1_data.txt','fold_2_data.txt','fold_3_data.txt','fold_4_data.txt']
    cache_file = os.path.join(cache_dir, 'adience.pth')

    if not os.path.isfile(cache_file):
        print "generating cache_file"
        image_paths, face_ids, ages, genders, locations, angs, yaws, scores=[],[],[],[],[],[],[],[]
        for txt in txt_list:
            txt_path = os.path.join(root_path, txt)
            with open(txt_path) as f:
                lines = f.readlines()[1:]
                for line in lines:
                    user_id, original_image, face_id, age, gender, x,\
                    y, dx, dy, tilt_ang, yaw, fiducial_score = line.split('\t')
                    age=eval(age)
                    if type(age)==tuple:
                        try:age=AGES_START.index(age[0])
                        except:
                            continue
                    elif type(age)==int:
                        age=np.argmin(np.abs(AGES_MID-age))
                    else:
                        continue  # include None
                    try: gender=GENDER.index(gender)
                    except:
                        continue
                    image_paths.append(os.path.join(user_id, original_image))
                    face_ids.append(int(face_id))
                    ages.append(age)
                    genders.append(gender)
                    locations.append((int(x),int(y),int(dx),int(dy)))
                    angs.append(int(tilt_ang))
                    yaws.append(int(yaw))
                    scores.append(int(fiducial_score))
        save_obj=[image_paths, face_ids, ages, genders, locations, angs, yaws, scores]
        pickle.dump(save_obj, open(cache_file, 'wb'))
    else:
        print "read from cache_file"
        image_paths, face_ids, ages, genders, locations, angs, yaws, scores = pickle.load(open(cache_file, 'rb'))
    print "read mat OK"
    return image_paths, face_ids, ages, genders, locations, angs, yaws, scores

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('root_path',help='root path of adience datasets',type=str)
    args=parser.parse_args()
    read_txt(args.root_path)
