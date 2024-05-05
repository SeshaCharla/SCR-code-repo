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
gsec2kgmin_gain = 1/16.6667   # Conversion factor from g/sec to kg/min


# Class to load the data -------------------------------------------------------
class data(object):
    def __init__(self, tt, age, num):
        # Variables
        self.raw = {'t':None, 'x1':None, 'x2':None,
                    'u1':None, 'u2':None, 'T':None, 'F':None, 'y1':None}
        self.ssd = {'t':None, 'x1':None, 'x2':None,
                    'u1':None, 'u2':None, 'T':None, 'F':None}
        self.iod = {'t':None, 'y1':None,
                    'u1':None, 'u2':None, 'T':None, 'F':None}

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

    def gen_ssd(self):
        # Generate the state space data
        rmNaNrows = lambda x: np.delete(x, [i for i in range(len(x))
                                            if np.any(np.isnan(x[i]))],
                                                                axis=0)
        ssd_tab = rmNaNrows(np.matrix([self.raw['t'],
                                       self.raw['x1'], self.raw['x2'],
                                       self.raw['u1'], self.raw['u2'],
                                       self.raw['T'],  self.raw['F']]).T)
        ssd_mat = ssd_tab.T
        self.ssd['t'] = np.array(ssd_mat[0]).flatten()
        self.ssd['x1'] = np.array(ssd_mat[1]).flatten()
        self.ssd['x2'] = np.array(ssd_mat[2]).flatten()
        self.ssd['u1'] = np.array(ssd_mat[3]).flatten()
        self.ssd['u2'] = np.array(ssd_mat[4]).flatten()
        self.ssd['T'] = np.array(ssd_mat[5]).flatten()
        self.ssd['F'] = np.array(ssd_mat[6]).flatten()


    def gen_iod(self):
        # Generate the input output data
        rmNaNrows = lambda x: np.delete(x, [i for i in range(len(x))
                                            if np.any(np.isnan(x[i]))],
                                                                axis=0)
        iod_tab = rmNaNrows(np.matrix([self.raw['t'],
                                       self.raw['y1'],
                                       self.raw['u1'], self.raw['u2'],
                                       self.raw['T'],  self.raw['F']]).T)
        if self.name == "mes_18": # Special case for mes_18
            iod_tab = np.copy(iod_tab[247:])
        iod_mat = iod_tab.T
        self.iod['t'] = np.array(iod_mat[0]).flatten()
        self.iod['y1'] = np.array(iod_mat[1]).flatten()
        self.iod['u1'] = np.array(iod_mat[2]).flatten()
        self.iod['u2'] = np.array(iod_mat[3]).flatten()
        self.iod['T'] = np.array(iod_mat[4]).flatten()
        self.iod['F'] = np.array(iod_mat[5]).flatten()


    def pickle_data(self):
        # Create a dictionary of the data
        data_dict = {'ssd':self.ssd, 'iod':self.iod, 'raw':self.raw}
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
        self.ssd = data_dict['ssd']
        self.iod = data_dict['iod']
        self.raw = data_dict['raw']


    def load_test_data(self):
        # Load the test data
        file_name = test_dir + "/" + test_dict[self.name]
        data = read_csv(file_name, header=[0,1])
        # Assigning the data to the variables
        self.raw['t'] = np.array(data.get(('LOG_TM', 'sec'))).flatten()
        self.raw['F'] = np.array(data.get(('EXHAUST_FLOW', 'kg/min'))).flatten()
        Tin = np.array(data.get(('V_AIM_TRC_DPF_OUT', 'Deg_C'))).flatten()
        Tout = np.array(data.get(('V_AIM_TRC_SCR_OUT', 'Deg_C'))).flatten()
        self.raw['T'] = np.mean([Tin, Tout], axis=0).flatten()
        self.raw['x1'] = np.array(data.get(('EXH_CW_NOX_COR_U1', 'PPM'))).flatten()
        self.raw['x2'] = np.array(data.get(('EXH_CW_AMMONIA_MEA', 'ppm'))).flatten()
        self.raw['y1'] = np.array(data.get(('V_SCM_PPM_SCR_OUT_NOX', 'ppm'))).flatten()
        self.raw['u1'] = np.array(data.get(('ENG_CW_NOX_FTIR_COR_U2', 'PPM'))).flatten()
        self.raw['u2'] = np.array(data.get(('V_UIM_FLM_ESTUREAINJRATE', 'ml/sec'))).flatten()
        # u1_sensor = np.array(data.get(('EONOX_COMP_VALUE', 'ppm'))).flatten()
        self.gen_ssd()
        self.gen_iod()
        self.pickle_data()


    def load_truck_data(self):
        # Load the truck data
        file_name = truck_dir + "/" + truck_dict[self.name]
        data = loadmat(file_name)
        # Assigning the data to the variables
        self.raw['t'] = np.array(data['tod']).flatten()
        self.raw['F'] = np.array(data['pExhMF']).flatten() * gsec2kgmin_gain
        self.raw['T'] = np.array(data['pSCRBedTemp']).flatten()
        self.raw['u2'] = np.array(data['pUreaDosing']).flatten()
        self.raw['u1'] = np.array(data['pNOxInppm']).flatten()
        self.raw['y1'] = np.array(data['pNOxOutppm']).flatten()
        self.gen_iod()
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

    # Plotting all the data sets
    import matplotlib.pyplot as plt

    for i in range(2):
        for j in range(3):
            for key in ['u1', 'u2', 'T', 'F', 'x1', 'x2']:
                plt.figure()
                plt.plot(test_data[i][j].ssd['t'], test_data[i][j].ssd[key], label=test_data[i][j].name + " " + key)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(test_data[i][j].name)
                plt.savefig("figs/" + test_data[i][j].name + "_" + key + ".png")
                plt.close()
            for key in ['u1', 'u2', 'T', 'F', 'y1']:
                plt.figure()
                plt.plot(test_data[i][j].iod['t'], test_data[i][j].iod[key], label=test_data[i][j].name + " " + key)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(test_data[i][j].name)
                plt.savefig("figs/" + test_data[i][j].name + "_" + key + ".png")
                plt.close()

    for i in range(2):
        for j in range(4):
            for key in ['u1', 'u2', 'T', 'F', 'y1']:
                plt.figure()
                plt.plot(truck_data[i][j].raw['t'], truck_data[i][j].raw[key], label=truck_data[i][j].name + " " + key)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(truck_data[i][j].name)
                plt.savefig("figs/" + truck_data[i][j].name + "_" + key + ".png")
                plt.close()

    # Showing datat discontinuities --------------------------------------------
    plt.figure()
    for i in range(2):
        for j in range(3):
            t = test_data[i][j].ssd['t']
            plt.plot(np.arange(len(t)), t, label=test_data[i][j].name + 'ss')
            t = test_data[i][j].iod['t']
            plt.plot(np.arange(len(t)), t, label=test_data[i][j].name + 'io')
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in test data')
    plt.savefig("figs/time_discontinuities_test.png")
    plt.close()

    plt.figure()
    for i in range(2):
        for j in range(4):
            t = truck_data[i][j].iod['t']
            plt.plot(np.arange(len(t)), t, label=truck_data[i][j].name)
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in truck data')
    plt.savefig("figs/time_discontinuities_truck.png")
    plt.close()
