def sample_get(datasource, cnt):
    X, Y = [], []
    for i in range(datasource.shape[0]):
        if i % cnt < seq_len - 1:
            continue
        tmpx, tmpy = [], []
        for j in range(seq_len):
            tmpx.append(datasource[i - seq_len - 1 + j, :-1])
            if j == seq_len - 1:
                tmpy.append(datasource[i - seq_len - 1 + j, -1])
        X.append(tmpx)
        Y.append(tmpy)
    return np.array(X), np.array(Y)

def sample_get_network(datasource, cnt):
    X = []
    for i in range(datasource.shape[0]):
        if i % cnt < seq_len - 1:
            continue
        tmpx = []
        for j in range(seq_len):
            tmpx.append(datasource[i - seq_len - 1 + j, :, :])
        X.append(tmpx)
    return np.array(X)

def sample_get_static(datasource, cnt):
    X = []
    for i in range(datasource.shape[0]):
        if i % cnt < seq_len - 1:
            continue
        X.append(datasource[i - seq_len - 1 + (seq_len - 1), :])
    return np.array(X)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mean_absolute_percentage_error_revise(y_true, y_pred):
    ma = label_max
    mi = label_min
    y_true = y_true * (ma - mi) + mi
    y_pred = y_pred * (ma - mi) + mi
    diff = diff = K.square(y_true - y_pred) / \
        K.clip(K.square(y_true), K.epsilon(), None)
    mean_lable_float32 = mean_label.astype(np.float32)
    return 10. * K.mean(diff, axis=-1) + loss_lambda / K.square(mean_lable_float32) * K.mean(K.square(y_pred - y_true), axis=-1)
    # return 10. * K.mean(diff, axis=-1)

def get_mape(y_true, y_pred, max_value, min_value):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df * (max_value - min_value) + min_value
    df_new = df[df.y_true > 10 - 1e-10]
    y_true = np.array(df_new.y_true)
    y_pred = np.array(df_new.y_pred)
    y_true_nofilter = np.array(df.y_true)
    y_pred_nofilter = np.array(df.y_pred)
    print('Number of sample whose label beyond 10: %d\n' % df_new.shape[0])
    res = sum(abs(2 * (y_true - y_pred) / (y_true + y_pred))) / len(y_true)
    res_2 = np.sqrt(np.mean((y_true - y_pred) * (y_true - y_pred)))
    res_3 = sum(abs((y_true - y_pred) / (y_true+10))) / len(y_true)
    fw=open('dmvst.npz','w')
    np.savez(fw, true=y_true, pred=y_pred)
    return res, res_2, res_3