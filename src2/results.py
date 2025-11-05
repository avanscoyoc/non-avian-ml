import pandas as pd
from pathlib import Path


def save_results(results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
