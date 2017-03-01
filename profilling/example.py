from layers import *
import csv

def CNN1(csv_save_path=None):
    # the example CNN net profilling
    # if csv_save_path was given, the xxx.csv file will be saved
    input=np.array([200,200,3])
    conv1=Conv(input,7,96,stride=2,pad=3,activation='relu')
    pool1=Pool(conv1(),3,pad=1,stride=2)
    conv2=Conv(pool1(),5,256,stride=2,pad=2,activation='relu')
    pool2=Pool(conv2(),3,pad=1,stride=2)
    conv3=Conv(pool2(),3,384,stride=1,pad=1,activation='relu')
    conv4_2=Conv(conv3(),3,256,stride=1,pad=1)
    rpn_conv33=Conv(conv4_2(),3,256,pad=1,stride=1)
    rpn_cls_score=Conv(rpn_conv33,1,18)
    rpn_bbox_pred=Conv(rpn_conv33,1,36)
    rois=np.array([60,60,256])
    roi_pool_conv5=Pool(rois,10,10)
    fc6_2=Fc(roi_pool_conv5,128)
    fc7_2=Fc(fc6_2,1024)
    cls_score=Fc(fc7_2,2)
    bbox_pred=Fc(fc7_2,8)

    if csv_save_path!=None:
        with open(csv_save_path,'w') as file:
            writer=csv.writer(file)
            writer.writerow(['name','input','output','dot','add','compare'])
            for layer in box:
                writer.writerow([layer.name,layer.input,layer.out,layer.dot,layer.add,layer.compare])

    for layer in box:
        print(','.join(str(j) for j in [layer.name,layer.input,layer.out,layer.dot,layer.add,layer.compare]))
CNN1()