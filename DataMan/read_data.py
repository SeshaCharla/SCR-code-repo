import numpy as np
from pandas import read_csv
from scipy.io import loadmat

# Data names for the truck and test data ---------------------------------------
# [0][j] - Degreened data
# [1][j] - Aged data
truck = [["adt_15", "mes_15", "wer_15", "trw_15"],
         ["adt_17", "mes_18", "wer_17", "trw_16"]]
test = [["dg_cftp", "dg_hftp", "dg_rmc"],
        ["aged_cftp", "aged_hftp", "aged_rmc"]]
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


# Class to load the data -------------------------------------------------------
class data:
    def __init__(self, tt, age, num):
        # Variables
        self.x1 = None
        self.x2 = None
        self.t = None
        self.y1 = None
        self.T = None
        self.F = None

        # Get the right data name and root directory
        if tt == "truck":
            self.name = truck[age][num]
            self.load_truck_data()
        elif tt == "test":
            self.name = test[age][num]
            self.load_test_data()
        else:
            raise(ValueError("Invalid data type"))

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
        self.u1_sensor = np.array(data.get(('EONOX_COMP_VALUE', 'ppm'))).flatten()

    def load_truck_data(self):
        # Load the truck data
        file_name = truck_dir + "/" + truck_dict[self.name]
        data = loadmat(file_name)
        # Assigning the data to the variables
        self.t = np.array(data['tod']).flatten()
        self.F = np.array(data['pExhMF']).flatten()
        self.T = np.array(data['pSCRBedTemp']).flatten()
        self.u2 = np.array(data['pUreaDosing']).flatten()
        self.u1 = np.array(data['pNOxInppm']).flatten()
        self.y1 = np.array(data['pNOxOutppm']).flatten()

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
    import matplotlib.pyplot as plt