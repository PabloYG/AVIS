import numpy as np
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from feature_matching import independent_matching

# sphinx_gallery_thumbnail_number = 2


v1_descriptor_list = []
v2_descriptor_list = []

v1_all_descriptors = []
v2_all_descriptors = []

v1_keypoint_list = []
v2_keypoint_list = []


def feature_detection(img, video_number):

    print("Frame Count: {}".format(len(v1_frameList)))

    # find the keypoints and descriptors with SIFT
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if video_number is "v1":
        v1_descriptor_list.append(descriptors)
        v1_keypoint_list.append(keypoints)
        v1_all_descriptors.extend(descriptors)
    elif video_number is "v2":
        v2_descriptor_list.append(descriptors)
        v2_keypoint_list.append(keypoints)
        v2_all_descriptors.extend(descriptors)

    # print("All Descriptors:")
    # print(v1_descriptor_list)
    # print("Descriptor List:")
    # print(v1_all_descriptors)
    print("Descriptor Count: {}".format(len(v1_descriptor_list)))


v1_frame_to_v2_sequence_matches = []


v2_frame_to_v1_sequence_matches = []


def feature_matching():

    # FLANN parameters
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # test_match_1 = flann.knnMatch(v1_descriptor_list[3], v2_descriptor_list[20], k=2)
    #
    # test_match_2 = flann.knnMatch(v1_descriptor_list[20], v2_descriptor_list[3], k=2)
    #
    # print("Test Match between v1[3] and v2[3]")
    # print(len(test_match_1))
    # print("Test Match between v1[3] and v2[20]")
    # print(len(test_match_2))

    matchesMask1 = [[0, 0] for i in range(len(v2_descriptor_list))]

    for j in range(len(v2_descriptor_list)):

        v2_frame_to_v1_frame_matches = []

        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

        bf.clear()
        clusters = np.array([v2_descriptor_list[j]])
        bf.add(clusters)
        bf.train()
        # print("Train Collection:")
        # print(bf.getTrainDescriptors())
        # print("V2 Descriptors:")
        # print(v2_descriptor_list[j])

        for i in range(len(v1_descriptor_list)):
            v2_matches = bf.knnMatch(v1_descriptor_list[i], k=2)
            good_matches = []
            good_keypoints_1 = []
            good_keypoints_2 = []
            iterable = 0
            for m, n in v2_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m][0])
                    good_keypoints_1.append(v1_keypoint_list[i][v2_matches[iterable][0].queryIdx].pt)
                    good_keypoints_2.append(v2_keypoint_list[j][v2_matches[iterable][0].trainIdx].pt)
                iterable += 1
            sum_of_descriptor_weights = 0
            sum_of_keypoint_weights = 0
            for g in range(len(good_matches)):
                sum_of_descriptor_weights += decaying_weighting_function_for_descriptors(good_matches[g].distance)
                sum_of_keypoint_weights += decaying_weighting_function_for_keypoints(good_keypoints_1[g],
                                                                                     good_keypoints_2[g])
            v2_frame_to_v1_frame_matches.append(sum_of_descriptor_weights * sum_of_keypoint_weights)

        bf.clear()
        print(bf.getTrainDescriptors())
        del bf

        v2_frame_to_v1_sequence_matches.append(v2_frame_to_v1_frame_matches)
        print(len(v2_frame_to_v1_sequence_matches))
        print("v2_frame_to_v1_sequence_matches:")
        print(v2_frame_to_v1_sequence_matches[j])

    for i in range(len(v1_descriptor_list)):

        v1_frame_to_v2_frame_matches = []

        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

        bf.clear()
        clusters = np.array([v1_descriptor_list[i]])
        bf.add(clusters)
        bf.train()
        print("Train Collection:")
        print(bf.getTrainDescriptors())
        print("V1 Descriptors:")
        print(v1_descriptor_list[i])

        for j in range(len(v2_descriptor_list)):

            v1_matches = bf.knnMatch(v2_descriptor_list[j], k=2)
            good_matches = []
            good_keypoints_1 = []
            good_keypoints_2 = []
            iterable = 0
            for m, n in v1_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m][0])
                    good_keypoints_1.append(v2_keypoint_list[j][v1_matches[iterable][0].queryIdx].pt)
                    good_keypoints_2.append(v1_keypoint_list[i][v1_matches[iterable][0].trainIdx].pt)
                iterable += 1
            sum_of_descriptor_weights = 0
            sum_of_keypoint_weights = 0
            for g in range(len(good_matches)):
                sum_of_descriptor_weights += decaying_weighting_function_for_descriptors(good_matches[g].distance)
                sum_of_keypoint_weights += decaying_weighting_function_for_keypoints(good_keypoints_1[g],
                                                                                     good_keypoints_2[g])
            v1_frame_to_v2_frame_matches.append(sum_of_descriptor_weights * sum_of_keypoint_weights)

        bf.clear()
        print(bf.getTrainDescriptors())
        del bf

        v1_frame_to_v2_sequence_matches.append(v1_frame_to_v2_frame_matches)
        print(len(v1_frame_to_v2_sequence_matches))
        print("v1_frame_to_v2_sequence_matches:")
        print(v1_frame_to_v2_sequence_matches[i])


    # print("v1_descriptor_list:")
    # print(v1_descriptor_list)
    # print("v2_descriptor_list:")
    # print(v2_descriptor_list)

    print("v1_frame_to_v2_sequence_matches:")
    print(v1_frame_to_v2_sequence_matches)
    print("v2_frame_to_v1_sequence_matches:")
    print(v2_frame_to_v1_sequence_matches)

    histogram_and_cost_matrix()


v1_frame_names = []
v2_frame_names = []

# total_matches = np.array([
#                     [8.0, 2.0, 2.0, 3.0, 1.0, 4.0, 1.0],
#                     [2.0, 9.0, 4.0, 1.0, 2.7, 8.0, 1.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]
#                     ])


descriptors_weighting_value = 1


def decaying_weighting_function_for_descriptors(distance):

    weight = np.exp(-descriptors_weighting_value * distance)

    return weight


keypoints_weighting_value = 2


def decaying_weighting_function_for_keypoints(query_image_keypoint, train_image_keypoint):

    print("Query Image KP:")
    print(query_image_keypoint)
    print("Train Image KP:")
    print(train_image_keypoint)

    tuple_subtraction = np.subtract(query_image_keypoint, train_image_keypoint)

    distance = np.linalg.norm(tuple_subtraction)

    weight = np.exp(-keypoints_weighting_value * distance)

    return weight


def histogram_and_cost_matrix():

    costs_list_1 = []
    sigma = 4

    for i in range(len(v2_frame_to_v1_sequence_matches)):

        costs_list_2 = []

        for j in range(len(v1_frame_to_v2_sequence_matches)):
            if max(v2_frame_to_v1_sequence_matches[i]) != 0:
                costs_list_2.append((1 - v2_frame_to_v1_sequence_matches[i][j] / max(
                    v2_frame_to_v1_sequence_matches[i])) ** sigma)
            else:
                costs_list_2.append(1)

        costs_list_1.append(costs_list_2)

        # costs_list[i][j] = cost_of_match_1

        # try:
        #
        #     if v1_frame_to_v2_sequence_matches[i][j] is not None:
        #
        #         cost_of_match_1 = (1 - v2_frame_to_v1_sequence_matches[i][j] / max(
        #             v2_frame_to_v1_sequence_matches[i])) ** sigma
        #
        #         cost_of_match_2 = (1 - v1_frame_to_v2_sequence_matches[j][i] / max(
        #             v1_frame_to_v2_sequence_matches[j])) ** sigma
        #
        #         costs_list[i][j] = cost_of_match_1 + cost_of_match_2
        #
        # except IndexError:
        #
        #     print("Out of range")

    # if video_number is 1:
    #     total_matches = np.array(v1_frame_to_v2_sequence_matches)
    # elif video_number is 2:
    #     total_matches = np.array(v2_frame_to_v1_sequence_matches)

    total_matches = np.array(costs_list_1)

    for i, item in enumerate(v1_frame_to_v2_sequence_matches):
        v1_frame_names.append("V1 Frame {}".format(i))

    for j, item in enumerate(v2_frame_to_v1_sequence_matches):
        v2_frame_names.append(("V2 Frame {}".format(j)))

    fig, ax = plt.subplots()
    im = ax.imshow(total_matches)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(v1_frame_names)))
    ax.set_yticks(np.arange(len(v2_frame_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(v1_frame_names)
    ax.set_yticklabels(v2_frame_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(v2_frame_names)):
    #     for j in range(len(v1_frame_names)):
    #         text = ax.text(j, i, total_matches[i, j],
    #                        ha="center", va="center", color="w")

    ax.set_title("Sequence to Sequence Alignment Cost Matrix")
    fig.tight_layout()
    plt.show()


# Capture video from file
v1_cap = cv2.VideoCapture('C:\\Users\\pablo\\Desktop\\PROJECT - AVIS\\Test Videos\\Edited\\Test_Video_13.mp4')
v2_cap = cv2.VideoCapture('C:\\Users\\pablo\\Desktop\\PROJECT - AVIS\\Test Videos\\Edited\\Test_Video_12.mp4')
v1_frameList = []
v2_frameList = []

while True:

    v1_ret, v1_frame = v1_cap.read()
    v2_ret, v2_frame = v2_cap.read()

    if v1_frame is not None:
        v1_frameList.append(v1_frame)
    if v2_frame is not None:
        v2_frameList.append(v2_frame)

    if v1_ret is True and v2_ret is True:

        v1_gray = cv2.cvtColor(v1_frame, cv2.COLOR_BGR2GRAY)
        v2_gray = cv2.cvtColor(v2_frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('v1_frame', v1_gray)
        cv2.imshow('v2_frame', v2_gray)

    if v1_ret is False and v2_ret is False:
        break

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

print("V1 Frame Count:")
print(len(v1_frameList))

print("V2 Frame Count:")
print(len(v2_frameList))

for i in v1_frameList:
    feature_detection(i, "v1")

for j in v2_frameList:
    feature_detection(j, "v2")

feature_matching()

v1_cap.release()
cv2.destroyAllWindows()






# print (v1_features)

# for i in v1_frameList:
#     for j in v2_frameList:
#         feature_detection_and_matching.detection_and_matching(i,j)
#
# for j in v2_frameList:
#     for i in v1_frameList:
#         feature_detection_and_matching.detection_and_matching(j,i)

