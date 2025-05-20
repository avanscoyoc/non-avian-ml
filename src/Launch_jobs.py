import itertools
import subprocess

training_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
species_list = ['coyote', 'bullfrog', 'engine', 'field_cricket',
                'human_vocal', 'pacific_chorus_frog', 'woodhouses_toad']              
model_names = ['vgg', 'resnet', 'mobilenet', 'birdnet']
random_seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for training_size, species, model_name in itertools.product(training_sizes, species_list, model_names):
    job_args = [
        '--model_name=' + model_name,
        '--training_size=' + str(training_size),
        '--species_list=' + species,
        '--datatype=data',
        '--random_seed=' + str(random_seed),
        '--n_folds =' + str(5)
    ]

    # Run the job in the cloud
    cmd = [
        'gcloud', 'beta', 'run', 'jobs', 'execute', 'run-train-job',
        '--region', 'us-central1',
        '--args'
    ] + job_args

    print(f"Launching job: {cmd}")
    subprocess.Popen(cmd)
