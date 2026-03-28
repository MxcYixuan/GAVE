import numpy as np
import pandas as pd

np.random.seed(42)

# Generate test traffic data
num_advertisers = 5
num_timesteps = 48  # 48 time steps per period
num_periods = 1

data = []
for period_idx in range(num_periods):
    for advertiser_idx in range(num_advertisers):
        budget = np.random.uniform(500, 2000)
        cpa = np.random.uniform(5, 15)
        category = np.random.randint(0, 3)

        for t in range(num_timesteps):
            # Number of auctions at this timestep
            num_auctions = np.random.randint(50, 500)

            for _ in range(num_auctions):
                row = {
                    'deliveryPeriodIndex': period_idx,
                    'advertiserNumber': advertiser_idx,
                    'advertiserCategoryIndex': category,
                    'budget': budget,
                    'CPAConstraint': cpa,
                    'timeStepIndex': t,
                    'pValue': np.random.uniform(0.1, 5.0),
                    'pValueSigma': np.random.uniform(0.01, 0.5),
                    'leastWinningCost': np.random.uniform(0.1, 3.0),
                    'isExposed': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'cost': np.random.uniform(0.1, 2.0),
                    'conversionAction': np.random.choice([0, 1], p=[0.8, 0.2]),
                }
                data.append(row)

df = pd.DataFrame(data)
df.to_csv('data/traffic/period-x.csv', index=False)
print(f"Generated test data: {len(df)} rows")
print(df.head())