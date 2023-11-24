from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from model import IncomeModel  # Replace with your actual model class
from data_processing import data_new  # Importing data_new for the 'data' parameter

# Assuming consumption_types are defined as follows:
consumption_types = ['fsddekc', 'fsddcal', 'fsddfi', 'fsddpot', 'fsddpro', 'fsddiro', 'fsddrae', 'fsdddmg',
                     'fsddc', 'fsddmag', 'fsddsod', 'fsddfas', 'fsddsug']

average_nrf = 0
average_pos = 0
average_neg = 0

LIM_VAL = 10000  # Define LIM_VAL or retrieve it from model parameters

def person_portrayal(agent):
    if agent is None:
        return

    if agent.income < (LIM_VAL / 26):
        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "r": 0.5,
            "Layer": 0,
            "Color": "blue"  # or some logic based on agent's properties
        }
        return portrayal
    else:
        return None
model_params = {
    "NIT_R": {
        "type": "Slider",
        "value": 0.1,
        "label": "NIT_R",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01
    },
    "LIM_VAL": {
        "type": "Slider",
        "value": 10000,
        "label": "LIM_VAL",
        "min": 10000,
        "max": 50000,
        "step": 1000
    }
}

canvas_element = CanvasGrid(person_portrayal, 20, 20, 500, 500)

chart_element = ChartModule(
    [{"Label": label, "Color": color} for label, color in
     [("Average NRF", "#FF0000"), ("Average POS", "#00FF00"), ("Average NEG", "#0000FF")]]
)

server = ModularServer(
    IncomeModel,
    [canvas_element, chart_element],
    "Income Model Visualization",
    {"data": data_new, "consumption_types": consumption_types}
)

if __name__ == "__main__":
    server.launch()