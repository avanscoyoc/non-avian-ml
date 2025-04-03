from utils import DataProcessor, Model

# Define paths and species list
datapath = "/workspaces/non-avian-ml-toy/non-avian_ML/audio"
species_list = ["coyote", "human_vocal"]

# Load Data
data_processor = DataProcessor(datapath, species_list, datatype="data") # use "data" for BirdNET "data_5s" for Perch
all_species_df = data_processor.load_data()

# Initialize and Train Model
model = Model(model_name="BirdNET") # use "BirdNET" or use "Perch"
model.train_and_evaluate(all_species_df, species_list=species_list, folds=2)