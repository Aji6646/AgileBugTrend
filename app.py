# app.py

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

import nltk

# Ensure necessary NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

import amrutha_diss_v6 as amrutha_module  # Make sure this is in the same folder
import os

# Load data
df = pd.read_csv('processed_bug_data.csv')
df['Created'] = pd.to_datetime(df['Created'])
df['Resolved'] = pd.to_datetime(df['Resolved'])
df['Resolution_Status'] = df['Resolved'].apply(lambda x: 'Closed' if pd.notnull(x) else 'Open')
df['Created_Month'] = df['Created'].dt.to_period('M').astype(str)

# Prepare data
X = df['Description']
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline = amrutha_module.make_pipeline(amrutha_module.TfidfVectorizer(), amrutha_module.LogisticRegression())
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'bug_severity_model.pkl')

# Load model
model = joblib.load('bug_severity_model.pkl')
df['Predicted_Severity'] = model.predict(df['Description'])

# Visualization
def create_visualizations(df):
    trend_fig = px.line(df.groupby('Created_Month').size().reset_index(name='Count'),
                        x='Created_Month', y='Count', title='Monthly Bug Trend')
    severity_fig = px.pie(df, names='Severity', title='Severity Distribution')
    resolution_fig = px.box(df[df['Resolution_Status'] == 'Closed'],
                            x='Severity', y='Resolution_Time', title='Resolution Time by Severity', log_y=True)
    component_fig = px.treemap(df, path=['Component/s'], title='Component-wise Distribution')
    priority_fig = px.histogram(df, x='Priority', title='Priority Distribution', color='Severity', barmode='group')
    return trend_fig, severity_fig, resolution_fig, component_fig, priority_fig

trend_fig, severity_fig, resolution_fig, component_fig, priority_fig = create_visualizations(df)

# Dash app
app = Dash(__name__)
server = app.server 
app.layout = html.Div([
    html.H1("Bug Analysis Dashboard", style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(figure=trend_fig, style={'width': '50%'}),
        dcc.Graph(figure=severity_fig, style={'width': '50%'})
    ], style={'display': 'flex'}),
    html.Div([
        dcc.Graph(figure=resolution_fig, style={'width': '50%'}),
        dcc.Graph(figure=component_fig, style={'width': '50%'})
    ], style={'display': 'flex'}),
    html.Div([
        dcc.Graph(figure=priority_fig, style={'width': '50%'})
    ], style={'display': 'flex'}),
    html.Div([
        html.H2("Predict Bug Severity", style={'textAlign': 'center'}),
        dcc.Textarea(id='description-input', placeholder='Enter bug description...', style={'width': '100%', 'height': 100}),
        html.Button('Predict', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
    ])
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('description-input', 'value')
)
def predict_severity(n_clicks, description):
    if n_clicks > 0 and description:
        pred = model.predict([description])[0]
        return f"🔮 Predicted Severity: {pred}"
    return ""

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
