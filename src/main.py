from utils import DataProcessor, Model

# Define paths and species list
datapath = "/workspaces/non-avian-ml-toy/non-avian_ML/audio"
datatype = "data"
species_list = ["bullfrog", "coyote"]
model_name = "BirdNET"

# Load Data
data_processor = DataProcessor(datapath, species_list)
all_species_df = data_processor.load_data()

# Initialize and Train Model
model = Model()
target_species = ["bullfrog", "coyote"]  # Change this as needed
model.train_and_evaluate(all_species_df, target_species)