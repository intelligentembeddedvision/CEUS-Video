
# CLASSES
#classes = {'FNH':0, 'HCC':1, 'HMG':2, 'METAHIPO':3,  'METAHIPER':4}
classes = {'FNH':0, 'HCC':1, 'HHE':2}
no_classes = len(classes)

# PATHS
#code_base_directory = "/media/Data/MY/My Projects/Medical/CEUS/3dcnn-lstm/"
#code_base_directory = "/media/cat/Backup/MY/My OneDrives/UPT/OneDrive - Universitatea Politehnica Timisoara/Cod/CEUS-Python/3dcnn-lstm/"
code_base_directory = "/home/cat/MY/Projects/CEUS-Video/"

#data_base_directory = "/media/Data/MY/My Databases/MEDICAL/CEUS/SYSU/DNN-Video/"
#data_base_directory = "/media/cat/Backup/MY/My OneDrives/IEV/OneDrive - Universitatea Politehnica Timisoara/Baze de date/CEUS/SYSU-CEUS-FLL/DNN-Video/"
data_base_directory = "/home/cat/MY/Datasets/DNN-Video/"

train_path = data_base_directory + "Training_set"
test_path  = data_base_directory + "Testing_set"

# FRAMES
D = 16   #New Depth size => Number of frames.
W = 112  #New Frame Width.
H = 112  #New Frame Height.
C = 3    #Number of channels.

# TRAINING
batch_size = 50
no_epochs = 30
learning_rate = 0.0001
#validation_split = 0.2
verbosity = 1

model = None

CNN_type  = 0
LSTM_type = 1

nweights_path = code_base_directory + "/Run-time/Weights/weights_FLL.h5"