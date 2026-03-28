import numpy as np
import pandas as pd

np.random.seed(42)

# Training data parameters
num_trajectories = 100
max_timesteps = 48

def generate_state(timestep, budget, remaining_budget, avg_bid, avg_lwc, avg_pv, avg_conv, avg_xi):
    """Generate 16-dimensional state tuple"""
    timeleft = (max_timesteps - timestep) / max_timesteps
    bgtleft = remaining_budget / budget if budget > 0 else 0
    state = (
        timeleft,                          # 0: time left ratio
        bgtleft,                           # 1: budget left ratio
        avg_bid,                           # 2: avg bid all
        avg_bid * 0.9,                     # 3: avg bid last 3
        avg_lwc,                           # 4: avg least winning cost all
        avg_pv,                            # 5: avg pValue all
        avg_conv,                          # 6: avg conversion all
        avg_xi,                            # 7: avg xi all
        avg_lwc * 0.85,                    # 8: avg least winning cost last 3
        avg_pv * 0.95,                     # 9: avg pValue last 3
        avg_conv * 0.9,                    # 10: avg conversion last 3
        avg_xi * 0.92,                     # 11: avg xi last 3
        avg_pv,                            # 12: pValue current
        np.random.randint(50, 500),        # 13: volume current
        np.random.randint(0, timestep * 100) if timestep > 0 else 0,  # 14: historical volume
        np.random.randint(50, 300)          # 15: last 3 volume
    )
    return state

data = []
for traj_idx in range(num_trajectories):
    budget = np.random.uniform(500, 2000)
    cpa_constraint = np.random.uniform(5, 15)
    advertiser_number = traj_idx % 10
    category = np.random.randint(0, 3)

    avg_bid = np.random.uniform(2, 8)
    avg_lwc = np.random.uniform(1, 4)
    avg_pv = np.random.uniform(0.5, 3)
    avg_conv = np.random.uniform(0.01, 0.1)
    avg_xi = np.random.uniform(0.001, 0.01)

    remaining_budget = budget
    total_cost = 0
    total_conversion = 0

    for t in range(max_timesteps):
        # Action: bid price
        action = cpa_constraint * avg_pv * np.random.uniform(0.8, 1.2)

        # Current state
        current_state = generate_state(t, budget, remaining_budget, avg_bid, avg_lwc, avg_pv, avg_conv, avg_xi)

        # Simulate reward (conversion) based on bid success
        bid_success_prob = min(action / (avg_lwc + 0.1), 1.0)
        if np.random.random() < bid_success_prob:
            reward = avg_conv * np.random.uniform(0.8, 1.2)
            cost = action * np.random.uniform(0.5, 1.0)
        else:
            reward = 0
            cost = 0

        total_cost += cost
        total_conversion += reward
        remaining_budget = max(0, remaining_budget - cost)

        # Done flag
        done = 1 if t == max_timesteps - 1 else 0

        # Next state
        next_state = generate_state(t + 1, budget, remaining_budget, avg_bid * 1.01, avg_lwc, avg_pv * 0.99, avg_conv * 0.98, avg_xi)

        row = {
            'deliveryPeriodIndex': traj_idx // 10,
            'advertiserNumber': advertiser_number,
            'advertiserCategoryIndex': category,
            'budget': budget,
            'CPAConstraint': cpa_constraint,
            'realAllCost': total_cost,
            'realAllConversion': total_conversion,
            'timeStepIndex': t,
            'state': str(current_state),
            'action': action,
            'reward': reward,
            'reward_continuous': reward * avg_pv,
            'done': done,
            'next_state': str(next_state)
        }
        data.append(row)

df = pd.DataFrame(data)
df.to_csv('data/trajectory/trajectory_data.csv', index=False)
print(f"Generated training data: {len(df)} rows, {num_trajectories} trajectories")
print(df.head())