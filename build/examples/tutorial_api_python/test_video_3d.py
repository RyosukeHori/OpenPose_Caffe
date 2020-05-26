import argparse
import logging
import sys
import time
import math
import cv2
import numpy as np
from openpose import pyopenpose as op

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import common

from lifting.prob_model import Prob3dPose
from lifting.My_draw import plot_pose

#=== command===
# python3 run_video_3d.py --video=./media/video_name.mp4

if __name__ == '__main__':
    t = time.time()
    fps_time = 0
    
    parser = argparse.ArgumentParser(description='pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    args = parser.parse_args()

    params = dict()
    params["model_folder"] = "../../../models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(args.video)
    print("OpenPose start")
    #cap = cv2.VideoCapture('/home/hori/openpose/examples/media/walking4.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret_val, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('./output/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))

    count = 1
    
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    if cap is None:
        print("Video Error")
        sys.exit(0)
    while cap.isOpened():
        ret_val, dst = cap.read()
        if ret_val == True:
           
            datum = op.Datum()
            datum.cvInputData = dst
            opWrapper.emplaceAndPop([datum])
            #print("frame{}".format(count))
            #print(datum.poseKeypoints)

            poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')
            
            pose_2d_mpiis = []
            visibilities = []
            standard_w = 640
            standard_h = 480
          # === For multiple people ===
            #for poseKeypoint in datum.poseKeypoints:
            #    pose_2d_mpii, visibility = common.MPIIPart.from_coco(poseKeypoint)
            #    pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
            #    #pose_2d_mpiis.append([(int(x), int(y)) for x, y in pose_2d_mpii])
            #    visibilities.append(visibility)
            
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(datum.poseKeypoints[0])
            pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
            #pose_2d_mpiis.append([(int(x), int(y)) for x, y in pose_2d_mpii])
                
            visibilities.append(visibility)

            pose_2d_mpiis = np.array(pose_2d_mpiis)
            visibilities = np.array(visibilities)
            transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
            pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

            for i, single_3d in enumerate(pose_3d):
                fig = plot_pose(single_3d)
            
            fig.savefig("./output/im3d.jpg")
            img = cv2.imread("./output/im3d.jpg")
            plt.close()

            out_video.write(img)
            cv2.imshow("Result", img)
            if cv2.waitKey(1) == 27:
                break
            count += 1
        else:
            print("Reading Finished")
            break
    print("Total Flame : {}".format(count))
    total = time.time() - t
    print("Total Time: {}".format(total))
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()
