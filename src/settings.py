IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\preprocessed'
CSV_FILE = r"C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular\adni_final.csv"
TABULAR_DATA_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular'

FEATURES = ['age', 'gender_numeric', 'education', 'APOE4',
            'FDG', 'AV45', 'TAU', 'PTAU', 'MMSE', 'label_numeric',
            'FDG_missing', 'TAU_missing', 'PTAU_missing', 'AV45_missing']

TARGET = 'label_numeric'

IMAGE_SIZE = (182, 218, 182)

TRAIN_SIZE = 1
VAL_SIZE = 1
TEST_SIZE = 1
