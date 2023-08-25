"""
The following is an import of PyTorch libraries.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import netCDF4 as nc
import model
from loaddata import newnorm, data_loader


# Determine if any GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Initialize Hyperparameters
dim_NN = 564
dim_NNout = 140

batch_size = 8
learning_rate = 1e-4
num_epochs = 1


## load mean and std for normalization
fm = np.load("Demodata/mean_avg.npz")
fs = np.load("Demodata/std_avg.npz")

Um = fm["U"]
Vm = fm["V"]
Tm = fm["T"]
DSEm = fm["DSE"]
NMm = fm["NM"]
NETDTm = fm["NETDT"]
Z3m = fm["Z3"]
RHOIm = fm["RHOI"]
PSm = fm["PS"]
latm = fm["lat"]
lonm = fm["lon"]
UTGWSPECm = fm["UTGWSPEC"]
VTGWSPECm = fm["VTGWSPEC"]

Us = fs["U"]
Vs = fs["V"]
Ts = fs["T"]
DSEs = fs["DSE"]
NMs = fs["NM"]
NETDTs = fs["NETDT"]
Z3s = fs["Z3"]
RHOIs = fs["RHOI"]
PSs = fs["PS"]
lats = fs["lat"]
lons = fs["lon"]
UTGWSPECs = fs["UTGWSPEC"]
VTGWSPECs = fs["VTGWSPEC"]


# Initialize the network and the Adam optimizer
GWnet = model.FullyConnected()

optimizer = torch.optim.Adam(GWnet.parameters(), lr=learning_rate)


s_list = list(range(5, 6))

for val in s_list:
    if val > 0:
        GWnet.load_state_dict(torch.load("./conv_torch.pth"))
        GWnet.eval()
    print("data loader iteration", val)
    filename = f"./Demodata/Demo_timestep_{str(val).zfill(3)}.nc"

    F = nc.Dataset(filename)
    PS = np.asarray(F["PS"][0, :, :])
    PS = newnorm(PS, PSm, PSs)

    Z3 = np.asarray(F["Z3"][0, :, :, :])
    Z3 = newnorm(Z3, Z3m, Z3s)

    U = np.asarray(F["U"][0, :, :, :])
    U = newnorm(U, Um, Us)

    V = np.asarray(F["V"][0, :, :, :])
    V = newnorm(V, Vm, Vs)

    T = np.asarray(F["T"][0, :, :, :])
    T = newnorm(T, Tm, Ts)

    lat = F["lat"]
    lat = newnorm(lat, latm, lats)

    lon = F["lon"]
    lon = newnorm(lon, lonm, lons)

    DSE = np.asarray(F["DSE"][0, :, :, :])
    DSE = newnorm(DSE, DSEm, DSEs)

    RHOI = np.asarray(F["RHOI"][0, :, :, :])
    RHOI = newnorm(RHOI, RHOIm, RHOIs)

    NETDT = np.asarray(F["NETDT"][0, :, :, :])
    NETDT = newnorm(NETDT, NETDTm, NETDTs)

    NM = np.asarray(F["NMBV"][0, :, :, :])
    NM = newnorm(NM, NMm, NMs)

    UTGWSPEC = np.asarray(F["UTGWSPEC"][0, :, :, :])
    UTGWSPEC = newnorm(UTGWSPEC, UTGWSPECm, UTGWSPECs)

    VTGWSPEC = np.asarray(F["VTGWSPEC"][0, :, :, :])
    VTGWSPEC = newnorm(VTGWSPEC, VTGWSPECm, VTGWSPECs)

    print("shape of PS", np.shape(PS))
    print("shape of Z3", np.shape(Z3))
    print("shape of U", np.shape(U))
    print("shape of V", np.shape(V))
    print("shape of T", np.shape(T))
    print("shape of DSE", np.shape(DSE))
    print("shape of RHOI", np.shape(RHOI))
    print("shape of NETDT", np.shape(NETDT))
    print("shape of NM", np.shape(NM))
    print("shape of UTGWSPEC", np.shape(UTGWSPEC))
    print("shape of VTGWSPEC", np.shape(VTGWSPEC))

    x_test, y_test = data_loader(
        U, V, T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC
    )

    print("shape of x_test", np.shape(x_test))
    print("shape of y_test", np.shape(y_test))

    data = model.MyDataset(X=x_test, Y=y_test)
    test_loader = DataLoader(data, batch_size=len(data), shuffle=False)
    print(test_loader)

    for batch, (X, Y) in enumerate(test_loader):
        print(np.shape(Y))
        pred = GWnet(X)
        truth = Y.cpu().detach().numpy()
        predict = pred.cpu().detach().numpy()

    print(np.corrcoef(truth.flatten(), predict.flatten())[0, 1])
    print("shape of truth ", np.shape(truth))
    print("shape of prediction", np.shape(predict))

    np.save("./pred_data_" + str(iter) + ".npy", predict)
