from mesa import Agent


class IndividualAgent(Agent):
    def __init__(self, unique_id, model, income, consumption_types, age_sex, BASE_BENEFIT,NIT_R, flat_benefit_amount = 0):
        super().__init__(unique_id, model)
        
        self.BASE_BENEFIT = BASE_BENEFIT
        self.NIT_R = NIT_R
        self.flat_benefit_amount = flat_benefit_amount
        self.current_step = 0
        self.income = income
        self.age_sex = age_sex
        

        # Initialize consumption as a dictionary of dictionaries
        self.consumption = {c_type: {'value': consumption_types[c_type], 'elasticity': 0, 'increase_amount': 0}
                            for c_type in consumption_types}
        self.income_updated = False
        self.net_benefit = 0
        self.income_percentage_change = 0

    def calculate_elasticity(self, consumption_type):
        # Fetch regression results for the specific consumption type and age_sex
        regression_results = self.model.regression_results[consumption_type][self.age_sex]
        elasticity = (regression_results['b']/ self.income)
        #print(f'Agent: {self.unique_id}, With B: {regression_results["b"]} for {consumption_type}')
        return elasticity

    def reset_income_update_status(self):
        self.income_updated = False

    def update_income(self):
        if self.flat_benefit_amount > 0:
            # Apply a flat benefit regardless of income
            original_income = self.income
            self.income += self.flat_benefit_amount
            self.net_benefit = self.flat_benefit_amount
            self.income_percentage_change = (self.flat_benefit_amount / original_income) * 100
            self.income_updated = True 
            
        else:
            if self.income < self.BASE_BENEFIT:
            # Existing logic for benefit based on BASE_BENEFIT and NIT_R
                original_income = self.income
                nit_benefit = (self.BASE_BENEFIT - self.income) * self.NIT_R
                self.income += nit_benefit
                self.net_benefit = nit_benefit
                self.income_percentage_change = (nit_benefit / original_income) * 100
                self.income_updated = True   
            

    def update_consumption(self):
        if not self.income_updated or self.income <= 0:
            return

        for consumption_type in self.consumption.keys():
            # Calculate elasticity for each consumption type
            properties = self.consumption[consumption_type]
            self.consumption[consumption_type]['elasticity'] = self.calculate_elasticity(consumption_type)
            elasticity = self.consumption[consumption_type]['elasticity']
            base_value = self.consumption[consumption_type]['value']
            percentage_change = self.income_percentage_change * elasticity
            increase_amount = (percentage_change / 100) * base_value

            # Update the increase amount and the new value of the consumption
            properties['increase_amount'] = increase_amount
            properties['value'] += increase_amount

    def step(self):
        self.current_step = self.model.schedule.steps
        self.reset_income_update_status()
        self.update_income()
        self.update_consumption()

    def get_agent_data(self):
        # Collect and return data relevant for each agent
        agent_data = {
            'basic_benefit': self.BASE_BENEFIT,
            'nit_rate': self.NIT_R,
            'flat_benefit_amount': self.flat_benefit_amount,
            'step': self.current_step,
            'unique_id': self.unique_id,
            'biweekly_income': self.income,
            'net_benefit': self.net_benefit,
            'income_percentage_change': self.income_percentage_change,
            'income_updated': self.income_updated,
            'age_sex': self.age_sex
            
        }
        for consumption_type, properties in self.consumption.items():
            agent_data[f'{consumption_type}'] = properties['value']
            agent_data[f'{consumption_type}_increase_amount'] = properties['increase_amount']
            agent_data[f'{consumption_type}_elasticity'] = properties['elasticity']
        return agent_data
