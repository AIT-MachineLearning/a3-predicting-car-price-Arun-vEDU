import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import dill

# Step 1: Load the pre-trained model and bin edges
with open('app.pkl', 'rb') as file:
     loaded_model, scaler = dill.load(file)

# Example DataFrame (replace this with the actual DataFrame you're using)
df = pd.DataFrame({
    'selling_price': [   29825.23572178,   128182.89725297,   547713.4287198,   2340327.8161826,
 10000000.00000001]  
})

# Step 2: Prepare the bin edges for the prediction
y_log = np.log(df["selling_price"])
binned_data, bin_edges = pd.cut(y_log, bins=4, retbins=True)
bin_edges_original = np.exp(bin_edges)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Car Price Prediction App"),
    
    # Input fields for user to input new data
    html.Label('Car Year:'),
    dcc.Input(id='input-year', type='number', value=2020),  # Default year is 2020
    
    html.Label('Horsepower (max power):'),
    dcc.Input(id='input-power', type='number', value=120),  # Default power is 120 HP
    
    # Button to trigger prediction
    html.Button('Predict Price Category', id='predict-button', n_clicks=0),
    
    # Output area
    html.H2("Predicted Price Range:"),
    html.Div(id='output-prediction')
])

# Callback function to handle user input and predict the price category
@app.callback(
    Output('output-prediction', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-year', 'value'), State('input-power', 'value')]
)
def predict_price(n_clicks, year, power):
    if n_clicks > 0:
        # Prepare the new data for prediction
        new_data = np.array([[year, power]])  # Replace with actual values
        
        # Step 3: Scale the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)  # Scale the new data
        
        # Step 4: Add the intercept term to the new data
        intercept_new_data = np.ones((new_data_scaled.shape[0], 1))  # Shape (m, 1)
        new_data_with_intercept = np.concatenate((intercept_new_data, new_data_scaled), axis=1)  # Shape (m, n + 1)
        
        # Step 5: Predict the class of the new data
        predicted_class_label = loaded_model.predict(new_data_with_intercept)
        
        # Map the predicted class label to the original price range
        predicted_class = predicted_class_label[0]  # Take the first (and only) value
        price_range = (bin_edges_original[predicted_class], bin_edges_original[predicted_class + 1])
        
        # Output the predicted price range
        return f"The predicted price range is: {price_range[0]:,.2f} to {price_range[1]:,.2f} USD"
    return ""

# Step 7: Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
