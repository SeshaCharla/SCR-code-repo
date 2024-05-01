import numpy as np
from pandas import read_csv
from scipy.io import loadmat
import pathlib as pth
import pickle as pkl


# Data names for the truck and test data ---------------------------------------
# [0][j] - Degreened data
# [1][j] - Aged data
truck = [["adt_15", "mes_15", "wer_15", "trw_15"],
         ["adt_17", "mes_18", "wer_17", "trw_16"]]
test = [["dg_cftp", "dg_hftp", "dg_rmc"],
        ["aged_cftp", "aged_hftp", "aged_rmc"]]
name_dict = {"truck": truck, "test": test}
data_dir = "../../Data"
test_dir = data_dir + "/test_cell_data"
truck_dir = data_dir + "/drive_data"
truck_dict = {"adt_15":"ADTransport_150814/ADTransport_150814_Day_File.mat",
              "adt_17":"ADTransport_170201/ADTransport_170201_dat_file.mat",
              "mes_15":"MesillaValley_150605/MesillaValley_150605_day_file.mat",
              "mes_18":"MesillaValley_180314/MesillaValley_180314_day_file.mat",
              "wer_15":"Werner_151111/Werner_151111_day_file.mat",
              "wer_17":"Werner_20171006/Werner_20171006_day_file.mat",
              "trw_15":"Transwest_150325/Transwest_150325_day_file.mat",
              "trw_16":"Transwest_161210/Transwest_161210_day_file.mat"}
test_dict = {"aged_cftp":"g580040_Aged_cFTP.csv",
             "aged_hftp":"g580041_Aged_hFTP.csv",
             "aged_rmc" :"g580043_Aged_RMC.csv",
             "dg_cftp"  :"g577670_DG_cFTP.csv",
             "dg_hftp"  :"g577671_DG_hFTP.csv",
             "dg_rmc"   :"g577673_DG_RMC.csv"}
gsec2kgmin_gain = 1/16.6667


# Class to load the data -------------------------------------------------------
class data(object):
    def __init__(self, tt, age, num):
        # Variables
        self.x1 = None
        self.x2 = None
        self.t = None
        self.y1 = None
        self.T = None
        self.F = None
        self.Medians = None
        self.Means = None

        # Get the right data name and root directory
        if tt == "truck":
            self.name = truck[age][num]
            try:
                self.load_pickle()
            except FileNotFoundError:
                self.load_truck_data()
        elif tt == "test":
            self.name = test[age][num]
            try:
                self.load_pickle()
            except FileNotFoundError:
                self.load_test_data()
        else:
            raise(ValueError("Invalid data type"))


    def pickle_data(self):
        # Create a dictionary of the data
        data_dict = {"x1":self.x1, "x2":self.x2, "t":self.t , 'u1':self.u1,
                     'u2':self.u2, "y1":self.y1, "T":self.T, "F":self.F,
                     "Medians":self.Medians, "Means":self.Means}
        # Pickle the data_dict to files
        pkl_file = pth.Path("./pkl_files/" + self.name + ".pkl")
        pkl_file.parent.mkdir(parents=True, exist_ok=True)
        with pkl_file.open("wb") as f:
            pkl.dump(data_dict, f)


    def load_pickle(self):
        # Load the pickled data
        pkl_file = pth.Path("./pkl_files/" + self.name + ".pkl")
        with pkl_file.open("rb") as f:
            data_dict = pkl.load(f)
        # Assign the data to the variables
        self.x1 = data_dict["x1"]
        self.x2 = data_dict["x2"]
        self.t = data_dict["t"]
        self.u1 = data_dict["u1"]
        self.u2 = data_dict["u2"]
        self.y1 = data_dict["y1"]
        self.T = data_dict["T"]
        self.F = data_dict["F"]
        self.Medians = data_dict["Medians"]
        self.Means = data_dict["Means"]


    def load_test_data(self):
        # Load the test data
        file_name = test_dir + "/" + test_dict[self.name]
        data = read_csv(file_name, header=[0,1])
        # Assigning the data to the variables
        self.t = np.array(data.get(('LOG_TM', 'sec'))).flatten()
        self.F = np.array(data.get(('EXHAUST_FLOW', 'kg/min'))).flatten()
        Tin = np.array(data.get(('V_AIM_TRC_DPF_OUT', 'Deg_C'))).flatten()
        Tout = np.array(data.get(('V_AIM_TRC_SCR_OUT', 'Deg_C'))).flatten()
        self.T = np.mean([Tin, Tout], axis=0).flatten()
        self.x1 = np.array(data.get(('EXH_CW_NOX_COR_U1', 'PPM'))).flatten()
        self.x2 = np.array(data.get(('EXH_CW_AMMONIA_MEA', 'ppm'))).flatten()
        self.y1 = np.array(data.get(('V_SCM_PPM_SCR_OUT_NOX', 'ppm'))).flatten()
        self.u1 = np.array(data.get(('ENG_CW_NOX_FTIR_COR_U2', 'PPM'))).flatten()
        self.u2 = np.array(data.get(('V_UIM_FLM_ESTUREAINJRATE', 'ml/sec'))).flatten()
        # self.u1_sensor = np.array(data.get(('EONOX_COMP_VALUE', 'ppm'))).flatten()
        self.Medians = {'x1':np.median(self.x1[~np.isnan(self.x1)]),
                        'x2':np.median(self.x2[~np.isnan(self.x2)]),
                        'u1':np.median(self.u1[~np.isnan(self.u1)]),
                        'u2':np.median(self.u2[~np.isnan(self.u2)]),
                        'T':np.median(self.T[~np.isnan(self.T)]),
                        'F':np.median(self.F[~np.isnan(self.F)]),
                        'y1':np.median(self.y1[~np.isnan(self.y1)])}
        self.Means = {'x1':np.mean(self.x1[~np.isnan(self.x1)]),
                      'x2':np.mean(self.x2[~np.isnan(self.x2)]),
                      'u1':np.mean(self.u1[~np.isnan(self.u1)]),
                      'u2':np.mean(self.u2[~np.isnan(self.u2)]),
                      'T':np.mean(self.T[~np.isnan(self.T)]),
                      'F':np.mean(self.F[~np.isnan(self.F)]),
                      'y1':np.mean(self.y1[~np.isnan(self.y1)])}
        self.pickle_data()


    def load_truck_data(self):
        # Load the truck data
        file_name = truck_dir + "/" + truck_dict[self.name]
        data = loadmat(file_name)
        # Assigning the data to the variables
        self.t = np.array(data['tod']).flatten()
        self.F = np.array(data['pExhMF']).flatten() * gsec2kgmin_gain
        self.T = np.array(data['pSCRBedTemp']).flatten()
        self.u2 = np.array(data['pUreaDosing']).flatten()
        self.u1 = np.array(data['pNOxInppm']).flatten()
        self.y1 = np.array(data['pNOxOutppm']).flatten()
        self.Medians = {'u1':np.median(self.u1[~np.isnan(self.u1)]),
                        'u2':np.median(self.u2[~np.isnan(self.u2)]),
                        'T':np.median(self.T[~np.isnan(self.T)]),
                        'F':np.median(self.F[~np.isnan(self.F)]),
                        'y1':np.median(self.y1[~np.isnan(self.y1)])}
        self.Means = {'u1':np.mean(self.u1[~np.isnan(self.u1)]),
                      'u2':np.mean(self.u2[~np.isnan(self.u2)]),
                      'T':np.mean(self.T[~np.isnan(self.T)]),
                      'F':np.mean(self.F[~np.isnan(self.F)]),
                      'y1':np.mean(self.y1[~np.isnan(self.y1)])}
        self.pickle_data()


#-------------------------------------------------------------------------------

# Functions to load the data sets ----------------------------------------------

def load_test_data_set():
    # Load the test data
    return [[data("test", age, tst) for tst in range(3)] for age in range(2)]


def load_truck_data_set():
    # Load the truck data
    return [[data("truck", age, tst) for tst in range(4)] for age in range(2)]



#-------------------------------------------------------------------------------

if __name__=="__main__":

    # Acutaly load the entire data set -----------------------------------------
    test_data = load_test_data_set()
    truck_data = load_truck_data_set()

    # Getting the medians of all the data points
    test_medians = {'x1':[], 'x2':[], 'u1':[], 'u2':[], 'F':[], 'T':[], 'y1':[]}
    test_means = {'x1':[], 'x2':[], 'u1':[], 'u2':[], 'F':[], 'T':[], 'y1':[]}
    for i in range(2):
        for j in range(3):
            for key in test_medians:
                test_medians[key].append(test_data[i][j].Medians[key])
                test_means[key].append(test_data[i][j].Means[key])
    for key in test_medians:
        test_medians[key] = np.array(test_medians[key]).flatten()
        test_means[key] = np.array(test_means[key]).flatten()

    truck_medians = {'u1':[], 'u2':[], 'F':[], 'T':[], 'y1':[]}
    truck_means = {'u1':[], 'u2':[], 'F':[], 'T':[], 'y1':[]}
    for i in range(2):
        for j in range(4):
            for key in truck_medians:
                truck_medians[key].append(truck_data[i][j].Medians[key])
                truck_means[key].append(truck_data[i][j].Means[key])
    for key in truck_medians:
        truck_medians[key] = np.array(truck_medians[key]).flatten()
        truck_means[key] = np.array(truck_means[key]).flatten()
