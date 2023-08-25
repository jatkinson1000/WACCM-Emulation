import numpy as np

dim_NN = int(564)
dim_NNout = int(140)


def newnorm(var, varm, varstd):
    dim = varm.size
    if dim > 1:
        vara = var - varm[:, np.newaxis, np.newaxis]
        tmp = vara / varstd[:, np.newaxis, np.newaxis]
    else:
        tmp = (var - varm) / varstd
    return tmp


def data_loader(U, V, T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC):
    Nlat = U.shape[1]
    Nlon = U.shape[2]
    Ncol = Nlat * Nlon

    x_train = np.zeros([dim_NN, Ncol])
    y_train = np.zeros([dim_NNout, Ncol])

    x_train[0:70, :] = U.reshape(70, Ncol)
    x_train[70 : 2 * 70, :] = V.reshape(70, Ncol)
    x_train[70 + 70 : 3 * 70, :] = T.reshape(70, Ncol)
    x_train[3 * 70 : 4 * 70, :] = DSE.reshape(70, Ncol)
    x_train[4 * 70 : 5 * 70, :] = NM.reshape(70, Ncol)
    x_train[5 * 70 : 6 * 70, :] = NETDT.reshape(70, Ncol)
    x_train[6 * 70 : 7 * 70, :] = Z3.reshape(70, Ncol)
    x_train[7 * 70 : 8 * 70 + 1, :] = RHOI.reshape(71, Ncol)
    x_train[8 * 70 + 1 : 8 * 70 + 2, :] = PS.reshape(1, Ncol)
    x_train[8 * 70 + 2 : 8 * 70 + 3, :] = np.transpose(np.tile(lat, (Nlon, 1))).reshape(
        1, Ncol
    )
    x_train[8 * 70 + 3 : 8 * 70 + 4, :] = np.tile(lon, (1, Nlat))

    y_train[0:70, :] = UTGWSPEC.reshape(70, Ncol)
    y_train[70 : 2 * 70, :] = VTGWSPEC.reshape(70, Ncol)

    return x_train, y_train
