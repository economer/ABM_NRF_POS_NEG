import pandas as pd
import numpy as np
from data_processing import data_new
from model import IncomeModel
#import random

def run_sim(num_steps, flat_benefit_amount=0):
    #random.seed(42)  # Set a fixed seed for random
    np.random.seed(40)
    mbm = 46432 / 26
    lim = 24263 / 26
    lic = 20386 / 26
    base_benefit_values = [mbm, lim, lic]  # Include all base benefit levels
    nit_r_values = [0.3, 0.5, 0.8, 1.0]
    consumption_types = ['pos', 'neg']
    
    data = data_new
    all_results = []

    # First, run the simulation with the flat benefit for all agents
    model_flat_benefit = IncomeModel(0, 0, data, consumption_types, flat_benefit_amount=flat_benefit_amount)
    for _ in range(num_steps):
        model_flat_benefit.step()
    flat_benefit_data = model_flat_benefit.datacollector.get_agent_vars_dataframe()
    flat_benefit_df = pd.DataFrame(flat_benefit_data['Agent Data'].apply(pd.Series)).reset_index()
    flat_benefit_df['scenario'] = 'flat_benefit'  # Mark these results as from the flat benefit scenario
    all_results.append(flat_benefit_df)

    # Then, run the simulation for different combinations of base_benefit and nit_r
    run_id = 0  # Initialize a run identifier
    for base_benefit in base_benefit_values:
        for nit_r in nit_r_values:
            model = IncomeModel(nit_r, base_benefit, data, consumption_types, flat_benefit_amount=0)
            for _ in range(num_steps):
                model.step()
            model_data = model.datacollector.get_agent_vars_dataframe()
            agent_data = model_data['Agent Data'].apply(pd.Series)
            df = pd.DataFrame(agent_data).reset_index()
            df['scenario'] = f'base_benefit_{base_benefit}_nit_r_{nit_r}'
            df['run_id'] = run_id  # Add run identifier
            all_results.append(df)
            run_id += 1 

    # Combine all results into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df

# Example usage
#df_results = run_sim(num_steps=10, flat_benefit_amount=500)

