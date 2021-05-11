import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import plotly.express as px

# read the case time series data
cases = pd.read_csv("data/case_time_series.csv")
cases['Case_Date'] = pd.to_datetime(cases['Date_YMD'], format='%Y-%m-%d')
# cases.head()

# lets remove the data for last 3 days so we can predict
cases.drop(cases.tail(3).index, inplace=True)
# cases.tail()

tests = pd.read_csv("data/tested_numbers_icmr_data.csv")
tests['Testing_Date'] = pd.to_datetime([i.split(' ', 1)[0] for i in tests['Update Time Stamp']],
                                       infer_datetime_format=True)
tests.tail()

# there are some duplicate rows in tests - drop them
tests.drop_duplicates(subset=['Testing_Date'], inplace=True)
tests.head()

tests.drop(tests.tail(3).index, inplace=True)
tests.dropna(subset=["Daily RTPCR Samples Collected_ICMR Application"], inplace=True)
tests['Daily_Tests'] = tests['Daily RTPCR Samples Collected_ICMR Application']

# lets join both by dates
result = pd.merge(cases[['Case_Date', 'Daily Confirmed', 'Daily Recovered']],
                  tests[['Testing_Date', 'Daily_Tests']],
                  left_on='Case_Date',
                  right_on='Testing_Date')
result.tail()

# lets plot using linear regression
fig1 = px.scatter(result, x='Daily_Tests', y='Daily Confirmed', trendline='ols')
# go.Figure(go.Scatter(x=result['Daily_Tests'], y=result['Daily Confirmed'], mode='markers', trendline='ols'))
fig1.update_layout(
    title='First prediction model',
    showlegend=True)

# now we do some modeling
model = LinearRegression()
features = ['Daily_Tests']
target = ['Daily Confirmed']

x_train = result[features]
y_train = result[target]

model.fit(x_train, y_train)

# tests for next 3 days = 934541, 580038, 1013002
# confirmed for next 3 days = 403808, 366455, 329491

# lets try to predict
# firstTestData = [[934541], [580038], [1013002]]
firstTestData = [[1013002]]
firstPrediction = model.predict(firstTestData)

# now check the mean absolute error
# firstActualResults = [403808, 366455, 329491]
firstActualResults = [329491]
maeOfFirstModel = mean_absolute_error(firstActualResults, firstPrediction)
# print("Mean absolute error of our model", mae)

# the second wave has behaved differently due to mutations and new strains
# we will limit the dataframe to after march this year
filteredResults = result[result['Case_Date'] >= '2021-03-01']
# filteredResults.shape

# plot
fig2 = px.scatter(filteredResults, x='Daily_Tests', y='Daily Confirmed', trendline='ols')
fig2.update_layout(
    title='Second prediction model',
    showlegend=True)

# redo the modeling
model = LinearRegression()

features = ['Daily_Tests']
target = ['Daily Confirmed']

model.fit(filteredResults[features], filteredResults[target])

# tests for next 3 days = 934541, 580038, 1013002
# confirmed for next 3 days = 403808, 366455, 329491

secondTestData = [[1013002]]
secondPrediction = model.predict(secondTestData)

secondActualResults = [329491]
maeOfSecondModel = mean_absolute_error(secondActualResults, secondPrediction)
# this is scary close

# Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Data modeling - Linear regression"

# Set up the layout
app.layout = html.Div(children=[
    html.H2('Data modeling for daily number of tests versus confirmed cases (India)'),
    html.Hr(),
    html.H5(f'Predicted cases via first model: {firstPrediction[0][0]}'),
    html.H5(f'Actual result for first model: {firstActualResults[0]}'),
    html.H5(f'MAE for first model: {maeOfFirstModel}'),
    dcc.Graph(
        id='assignment7.1',
        figure=fig1
    ),
    html.Hr(),
    html.H6('Error seems too much. This can be since the second wave has behaved differently due to mutations and new '
            'strains. We will limit the dataframe to after march this year and see'),
    html.H5(f'Predicted cases via second model: {secondPrediction[0][0]}'),
    html.H5(f'Actual result for second model: {secondActualResults[0]}'),
    html.H5(f'MAE for second model: {maeOfSecondModel}. This is scary close!!'),
    dcc.Graph(
        id='assignment7.2',
        figure=fig2
    ),
    html.A('Code on Github', href="https://github.com/pinaki-das-sage/assignment7"),
]
)

if __name__ == '__main__':
    app.run_server(debug=True)
