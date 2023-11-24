
# import libs
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agent import IndividualAgent



class IncomeModel(Model):
    def __init__(self, BASE_BENEFIT, NIT_R, data, consumption_types,flat_benefit_amount=0):
        self.schedule = RandomActivation(self)  
        self.NIT_R = NIT_R  
        self.BASE_BENEFIT = BASE_BENEFIT
        self.flat_benefit_amount = flat_benefit_amount
        self.data = data.copy()
        self.consumption_types = consumption_types
        self.regression_results = {c_type: {} for c_type in consumption_types}
        self.datacollector = DataCollector(
            agent_reporters={"Agent Data": lambda a: a.get_agent_data()}
        )

        for index, row in self.data.iterrows():
            # Create a dictionary for consumption data for each agent
            consumption_data = {c_type: row[c_type] for c_type in consumption_types}
            agent = IndividualAgent(row['caseid'], self, row['biweekly_income'], consumption_data, row['age_sex'], self.NIT_R, self.BASE_BENEFIT, self.flat_benefit_amount)
            self.schedule.add(agent)

        # Collect data immediately after initialization
        self.datacollector.collect(self)   


    def update_agent_data(self):
        # Update model data with current agent data
        current_data = pd.DataFrame([agent.get_agent_data() for agent in self.schedule.agents])
        current_data['inv_biweekly_income'] = current_data['biweekly_income'].apply(lambda x: 1 / x if x != 0 else 0)

       # print("Current DataFrame columns:", current_data.columns)  # Debug print to check columns
        self.data = current_data   

    def calculate_regression_results(self, consumption_type):
        # Ensure that the updated data is used for regression
        grouped_data = self.data.groupby('age_sex')
        for name, group in grouped_data:
            model = smf.ols(f"({consumption_type}) ~ np.log(biweekly_income)", data=group).fit()
            
            if model.pvalues['np.log(biweekly_income)'] < 0.1:
                b = model.params['np.log(biweekly_income)'] 
            else:
                b = 0    
        

            self.regression_results[consumption_type][name] = {
                'a': model.params['Intercept'],
                'b': b
            }

    def step(self):
        # Update the agent data
        self.update_agent_data()

        # Calculate regression results for each consumption type
        for consumption_type in self.consumption_types:
            self.calculate_regression_results(consumption_type)

        # Proceed with the regular step process
        self.schedule.step()
        self.datacollector.collect(self)
