"""Step 1: Stratified-sample 30 models from the final analysis dataset for manual sanity check.

Sampling design:
  - 15 popular  (top-10% by downloads in the 10-10-80 main analysis scheme)
  - 15 unpopular (bottom-80% by downloads in the same scheme)
  - random_state = 42 for reproducibility

Output: sampled_models.csv with id / downloads / likes / category.
"""

import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(script_dir, '..', 'filtered_data',
                         'filtered_model_data_by_downloads_10-10-80.csv')
OUTPUT_CSV = os.path.join(script_dir, 'sampled_models.csv')
N_PER_TIER = 15
SEED = 42


def main():
    df = pd.read_csv(INPUT_CSV, usecols=['id', 'downloads', 'likes', 'category'])
    print(f"loaded {len(df)} filtered models")

    popular = df[df['category'] == 'popular'].sample(n=N_PER_TIER, random_state=SEED)
    unpopular = df[df['category'] == 'unpopular'].sample(n=N_PER_TIER, random_state=SEED)

    sample = pd.concat([popular, unpopular]).reset_index(drop=True)
    sample.to_csv(OUTPUT_CSV, index=False)
    print(f"saved {len(sample)} sampled models -> {OUTPUT_CSV}")
    print(sample['category'].value_counts().to_string())


if __name__ == '__main__':
    main()
