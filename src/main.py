from utils import DataProcessor, Model

# Define paths and species list
datapath = "/workspaces/non-avian-ml-toy/non-avian_ML/audio"
datatype = "data_5s"
species_list = ["bullfrog", "woodhouses_toad","pacific_chorus_frog"]
model_name = "Perch"

# Load Data
data_processor = DataProcessor(datapath, species_list, datatype="data_5s")
all_species_df = data_processor.load_data()

# Initialize and Train Model
model = Model()
model.train_and_evaluate(all_species_df, species_list=species_list)