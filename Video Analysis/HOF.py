import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

X_train, Y_train, X_test, Y_test = ([] for i in range(4))
dim = (150, 150)


def training():
    for each_folder in tqdm(os.listdir("UCF3/Training/")):
        path = os.path.join("UCF3/Training/", each_folder)

        for vid in tqdm(os.listdir(path + "/")):
            vidpath = os.path.join(path, vid)


            cap = cv.VideoCapture(vidpath)
            ret, frame1 = cap.read()
            nframe1 = cv.resize(frame1, dim)
            prvs = cv.cvtColor(nframe1, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(nframe1)
            hsv[..., 1] = 255

            while (1):
                ret, frame2 = cap.read()
                if ret == False:
                    break
                All_histogram_bins = []
                if each_folder == "Jumping":
                    Y_train.append(1)
                elif each_folder == "Tennis":
                    Y_train.append(2)
                else:
                    Y_train.append(3)

                nframe2 = cv.resize(frame2, dim)
                next = cv.cvtColor(nframe2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

                bin = ang * 180 / np.pi / 2

                Angles_blocks = []
                mag_blocks = []

                cellx = celly = 8

                histo_bins = dict({0: 0, 20: 0, 40: 0, 60: 0, 80: 0, 100: 0, 120: 0, 140: 0, 160: 0})
                for i in range(0, int(nframe2.shape[0] / celly)):
                    for j in range(0, int(nframe2.shape[1] / cellx)):
                        Angles_blocks.append(bin[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
                        mag_blocks.append(hsv[..., 2][i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
                # maxmm=max([sublist for sublist in Angles_blocks])
                for i in range(0, len(mag_blocks)):
                    for j in range(0, len(mag_blocks[i])):
                        for k in range(0, len(mag_blocks[j])):

                            if Angles_blocks[i][j][k] > 160 and Angles_blocks[i][j][k] < 180:
                                histo_bins[160] += ((180 - Angles_blocks[i][j][k]) / 20) * mag_blocks[i][j][k]
                                histo_bins[0] += ((Angles_blocks[i][j][k] - 160) / 20) * mag_blocks[i][j][k]

                            elif Angles_blocks[i][j][k] in histo_bins.keys():
                                histo_bins[Angles_blocks[i][j][k]] += mag_blocks[i][j][k]
                            elif Angles_blocks[i][j][k] == 180:
                                histo_bins[0] += mag_blocks[i][j][k]

                            else:
                                small_index = 10  # random assign
                                for d_key in histo_bins.keys():
                                    if d_key < Angles_blocks[i][j][k]:
                                        small_index = d_key
                                    else:
                                        break

                                histo_bins[small_index] += (((small_index + 20) - Angles_blocks[i][j][k]) / 20) * \
                                                           mag_blocks[i][j][k]
                                histo_bins[small_index + 20] += ((Angles_blocks[i][j][k] - small_index) / 20) * \
                                                                mag_blocks[i][j][k]

                    bins_vals = np.fromiter(histo_bins.values(), dtype=float)
                    bins_vals = bins_vals.reshape(9, 1)
                    All_histogram_bins.append(bins_vals)

                hist_array = np.asarray(All_histogram_bins)
                hist_array = hist_array.reshape(18, 18, 9, 1)

                hist_array = normalization(hist_array)

                features = np.hstack(hist_array)
                features = features.reshape(2916, 1)

                X_train.append(features)

                bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                # cv.imshow('frame2', bgr)
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv.imwrite('opticalfb.png', frame2)
                    cv.imwrite('opticalhsv.png', bgr)
                prvs = next
            cap.release()
            cv.destroyAllWindows()
    np.save('x_train.npy', X_train)
    np.save('y_train.npy', Y_train)
    return X_train, Y_train


def testing():
    for each_folder in tqdm(os.listdir("UCF3/Testing/")):
        path = os.path.join("UCF3/Testing/", each_folder)

        for vid in tqdm(os.listdir(path + "/")):
            vidpath = os.path.join(path, vid)

            cap = cv.VideoCapture(vidpath)
            ret, frame1 = cap.read()

            new_frame1 = cv.resize(frame1, dim)
            prvs = cv.cvtColor(new_frame1, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(new_frame1)
            hsv[..., 1] = 255

            while (1):
                ret, frame2 = cap.read()
                if ret == False:
                    break
                All_histogram_bins = []

                if each_folder == "Jumping":
                    Y_test.append(1)
                elif each_folder == "Tennis":
                    Y_test.append(2)
                else:
                    Y_test.append(3)
                new_frame2 = cv.resize(frame2, dim)
                next = cv.cvtColor(new_frame2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

                bin = ang * 180 / np.pi / 2

                Angles_blocks = []
                mag_blocks = []

                cellx = celly = 8

                histo_bins = dict({0: 0, 20: 0, 40: 0, 60: 0, 80: 0, 100: 0, 120: 0, 140: 0, 160: 0})
                for i in range(0, int(new_frame2.shape[0] / celly)):
                    for j in range(0, int(new_frame2.shape[1] / cellx)):
                        Angles_blocks.append(bin[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
                        mag_blocks.append(hsv[..., 2][i * celly: i * celly + celly, j * cellx: j * cellx + cellx])

                for i in range(0, len(mag_blocks)):
                    for j in range(0, len(mag_blocks[i])):
                        for k in range(0, len(mag_blocks[j])):

                            if Angles_blocks[i][j][k] > 160 and Angles_blocks[i][j][k] < 180:
                                histo_bins[160] += ((180 - Angles_blocks[i][j][k]) / 20) * mag_blocks[i][j][k]
                                histo_bins[0] += ((Angles_blocks[i][j][k] - 160) / 20) * mag_blocks[i][j][k]

                            elif Angles_blocks[i][j][k] in histo_bins.keys():
                                histo_bins[Angles_blocks[i][j][k]] += mag_blocks[i][j][k]
                            elif Angles_blocks[i][j][k] == 180:
                                histo_bins[0] += mag_blocks[i][j][k]
                            else:
                                small_index = 10  # random assign
                                for d_key in histo_bins.keys():
                                    if d_key < Angles_blocks[i][j][k]:
                                        small_index = d_key
                                    else:
                                        break

                                histo_bins[small_index] += ((histo_bins[small_index + 20] - Angles_blocks[i][j][
                                    k]) / 20) * \
                                                           mag_blocks[i][j][k]
                                histo_bins[small_index + 20] += ((Angles_blocks[i][j][k] - histo_bins[
                                    small_index]) / 20) * \
                                                                mag_blocks[i][j][k]

                    bins_vals = np.fromiter(histo_bins.values(), dtype=float)
                    bins_vals = bins_vals.reshape(9, 1)
                    All_histogram_bins.append(bins_vals)

                hist_array = np.asarray(All_histogram_bins)
                hist_array = hist_array.reshape(18, 18, 9, 1)

                hist_array = normalization(hist_array)

                features = np.hstack(hist_array)
                features = features.reshape(2916, 1)

                X_test.append(features)

                bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                # cv.imshow('frame2', bgr)
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv.imwrite('opticalfb.png', frame2)
                    cv.imwrite('opticalhsv.png', bgr)
                prvs = next
            cap.release()
            cv.destroyAllWindows()
    np.save('x_test.npy', X_test)
    np.save('y_test.npy', Y_test)
    return X_test, Y_test


def normalization(hist_array):
    for row in range(0, int((hist_array.shape[0]) - 1)):
        for col in range(0, int(hist_array.shape[1] - 1)):
            window = hist_array[row:row + 2, col:col + 2]
            norm_val = np.linalg.norm(window)
            hist_array[col:col + 2, col:col + 2] = window / norm_val
    return hist_array


def knn(XX_train, YY_train, XX_test, YY_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(XX_train, YY_train)

    predictions = knn.predict(XX_test)
    acc = accuracy_score(YY_test, predictions)

    print("KNN Accuracy n=3: ", acc)
    print(predictions.shape)
    calc_confusion(YY_test, predictions, "KNN")


def svm(XX_train, YY_train, XX_test, YY_test):
    ######## linear kernal#########
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(XX_train, np.ravel(YY_train))
    y_pred = svclassifier.predict(XX_test)
    calc_confusion(YY_test, y_pred, "linear")
    print(accuracy_score(np.ravel(YY_test), y_pred))

    '''svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(XX_train, np.ravel(YY_train))
    y_pred = svclassifier.predict(XX_test)
    confusion_matrix(YY_test, y_pred, "poly")

    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(XX_train, np.ravel(YY_train))
    y_pred = svclassifier.predict(XX_test)
    confusion_matrix(np.ravel(YY_test), y_pred, "rbf")

    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(XX_train, np.ravel(YY_train))
    y_pred = svclassifier.predict(XX_test)
    confusion_matrix(np.ravel(YY_test), y_pred, "sigmoid")'''
    return


def calc_confusion(ytest, predictions, typeofclassifier):
    print("Confusion matrix of" + typeofclassifier + ": ")
    print(confusion_matrix(ytest, predictions))


def main():
    # training()
    # testing()
    if os.path.exists('x_train.npy') and os.path.exists('y_train.npy'):
        X_train = np.load('x_train.npy').tolist()
        Y_train = np.load('y_train.npy').tolist()
    else:
        X_train, Y_train = training()

    XX_train = np.asarray(X_train)
    nsamples, nx, ny = XX_train.shape
    XX_train = XX_train.reshape((nsamples, nx * ny))

    print(XX_train.shape)
    # X_train=X_train.reshape()
    YY_train = np.asarray(Y_train)
    YY_train = YY_train.reshape(YY_train.shape[0], 1)
    print(YY_train.shape)

    if os.path.exists('x_test.npy') and os.path.exists('y_test.npy'):  # If you have already created the dataset:
        X_test = np.load('x_test.npy').tolist()
        Y_test = np.load('y_test.npy').tolist()
    else:
        X_test, Y_test = testing()

    XX_test = np.asarray(X_test)
    nsamples, nx, ny = XX_test.shape
    XX_test = XX_test.reshape((nsamples, nx * ny))
    YY_test = np.asarray(Y_test)
    YY_test = YY_test.reshape(YY_test.shape[0], 1)

    print(XX_test.shape)
    print(YY_test.shape)
    XX_train = np.nan_to_num(XX_train)
    XX_test = np.nan_to_num(XX_test)
    knn(XX_train, YY_train, XX_test, YY_test)
    svm(XX_train, YY_train, XX_test, YY_test)


main()




