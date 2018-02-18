from model import *
from utils import *

if __name__=='__main__':

    topo = {}
    fbojtopo = open('line_32.txt')
    for eachline in fbojtopo:
        t=eachline.strip().split('\t')
        topo[t[0]]=map(float, t[1:])
    fbojtopo.close()

    rawdata = pickle.load(open('dataset_20170201_20170326'))

    networkalltrain, networkalltest = [], []
    traindataall, testdataall = [], []
    trainyall, testyall = [], []
    staticidtrain, staticidtest = [], []
    topologytrain, topologytest = [], []

    mean_label = np.mean(traindataall[:, -1])
    scaler = MinMaxScaler(feature_range=(0, 1))

    traindataall = scaler.fit_transform(traindataall)
    testdataall = scaler.transform(testdataall)

    label_min = scaler.data_min_[-1]
    label_max = scaler.data_max_[-1]
    print label_min, label_max
    X_train, Y_train = sample_get(traindataall, traindataall.shape[0] / cnt)
    filter_train = np.nonzero(((Y_train[:, -1]) * (label_max - label_min)+label_min) > 10 - 1e-10)[0]
    X_train = X_train[filter_train, :, :]
    Y_train = Y_train[filter_train, :]
    print X_train.shape, Y_train.shape, np.max(Y_train)
    X_test, Y_test = sample_get(testdataall, testdataall.shape[0] / cnt)
    filter_test = np.nonzero(((Y_test[:, -1]) * (label_max - label_min)+label_min) > - 1e-10)[0]
    X_test = X_test[filter_test, :, :]
    Y_test = Y_test[filter_test, :]
    print X_test.shape, Y_test.shape, np.max(Y_test)


    image_train = sample_get_network(
        networkalltrain, networkalltrain.shape[0] / cnt)
    image_test = sample_get_network(
        networkalltest, networkalltest.shape[0] / cnt)
    image_train = np.reshape(image_train, tuple(list(image_train.shape) + [1]))
    image_train = image_train[filter_train, :, :, :, :]
    image_test = np.reshape(image_test, tuple(list(image_test.shape) + [1]))
    image_test = image_test[filter_test, :, :, :, :]
    print image_train.shape, image_test.shape

    staticidtrain = sample_get_static(staticidtrain, staticidtrain.shape[0] / cnt)
    staticidtest = sample_get_static(staticidtest, staticidtest.shape[0] / cnt)
    print staticidtrain.shape, staticidtest.shape

    staticidtrain = staticidtrain.astype(np.float32)
    staticidtest = staticidtest.astype(np.float32)

    topo_train = sample_get_static(topologytrain, topologytrain.shape[0] / cnt)
    topo_train = topo_train[filter_train, :]
    topo_test = sample_get_static(topologytest, topologytest.shape[0] / cnt)
    topo_test = topo_test[filter_test, :]
    print topo_train.shape, topo_test.shape



    model = build_model(X_train, Y_train, X_test, Y_test,
                        image_train, image_test, topo_train, topo_test, feature_len)
    # model = load_model('local_conv_lstm.h5')

    score = model.evaluate([image_test, X_test, topo_test], Y_test,
                           batch_size=batch_size, verbose=0)
    print "Test mse (norm): %.6f mse (real): %.6f " % (
        score[0], score[0] * (label_max - label_min) * (label_max - label_min))
    Y_pred = model.predict([image_test, X_test, topo_test],
                           batch_size=batch_size, verbose=0)
    print get_mape(Y_test.flatten(), Y_pred.flatten(), label_max, label_min)