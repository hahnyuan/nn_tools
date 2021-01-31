import cv2
import h5py
import numpy as np
import cPickle as pickle
import os
import time

class WiderDataset():
    annotation_names=['blur','pose','occlusion',
                     'invalid','expression','illumination']
    def __init__(self,path,set='val'):
        self.set=set
        self._data_path=path
        self.cache_file=os.path.join(self._data_path,'cache',self.set+'.pth')
        if not self._load_cache():
            self.file_names,self.boxes,self.annotations=self._read_data()
            pickle.dump([self.file_names,self.boxes,self.annotations],open(self.cache_file,'w'))
        self.len=len(self.file_names)

    def _show_pic_boxes(self,im,boxes,delay=1000):
        for box in boxes:
            x0,y0,x1,y1=[int(i) for i in np.round(box)]
            cv2.rectangle(im,(x0,y0),(x1,y1),(255,0,0))
        cv2.imshow("draw_pic", im)
        cv2.waitKey(delay)

    def show_pic_boxes(self,idx):
        im=cv2.imread(self.file_names[idx])
        self._show_pic_boxes(im,self.boxes[idx])


    def _load_cache(self):
        if os.path.exists(self.cache_file):
            print 'load from cache file',self.cache_file
            st_time=time.time()
            self.file_names, self.boxes, self.annotations =pickle.load(open(self.cache_file))
            print time.time()-st_time,'s used'
            return True
        return False

    def _read_data(self):
        mat_path=self._data_path + '/wider_face_split/wider_face_%s.mat'%(self.set,)
        print 'load from mat file', mat_path
        st_time = time.time()
        f = h5py.File(mat_path)
        # f=sio.loadmat(self._data_path + '/wider_face_split/wider_face_val.mat')
        annotations = {'blur':[],'pose':[],'occlusion':[],
                         'invalid':[],'expression':[],'illumination':[]}
        boxes=[]
        file_names = []
        for name in self.annotation_names:
            for folder in f[name+'_label_list'][0]:
                for image in f[folder][0]:
                    for value in f[image]:
                        annotations[name].append(value)
        for folder in f['face_bbx_list'][0]:
            for im_file in f[folder][0]:
                x = np.array([bbxs for bbxs in f[im_file]])
                x = x.transpose()
                x[:, 2:] += x[:, :2]
                boxes.append(x)
        for folder in f['file_list'][0]:
            for im_file in f[folder][0]:
                s = "".join([chr(c) for c in f[im_file]])
                file_names.append(self._data_path + '/images_no_fold/' + s + '.jpg')
        print time.time() - st_time, 's used'
        return file_names, boxes, annotations

    def sift_hard(self,idx,min_area=240,blur=True,pose=True,occlusion=True,invalid=True):
        # area = 0 for no area sift
        reserve=np.ones(len(self.boxes[idx]))
        if min_area:
            area=np.prod(self.boxes[idx][:,2:]-self.boxes[idx][:,:2],1)
            # print area
            reserve[area < min_area] = 0
        # if max_area:
        #     area = np.prod(self.boxes[idx][:, 2:] - self.boxes[idx][:, :2], 1)
        #     reserve[area > max_area] = 0
        if blur:
            reserve[self.annotations['blur'][idx]!=0]=0
        if pose:
            reserve[self.annotations['pose'][idx]!=0]=0
        if occlusion:
            reserve[self.annotations['occlusion'][idx]!=0]=0
        if invalid:
            reserve[self.annotations['invalid'][idx] != 0] = 0
        return reserve

    def get_idx_annotation(self,idx):
        annotation={}
        for name in self.annotation_names:
            annotation[name]=self.annotations[name][idx]
        return annotation

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        # given a idx then read the idxth image in wider face dataset
        # return [a numpy image with [1,height,width,BGR],boxes array ,annotations dict]
        im = cv2.imread(self.file_names[idx])
        while im is None:
            idx-=1
            im = cv2.imread(self.file_names[idx])
        boxes=self.boxes[idx]
        annotation=self.get_idx_annotation(idx)
        return im,boxes,annotation