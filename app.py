import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        AdOps Dashboard
    </h1>
    <p style='text-align: center; color: gray; font-size: 18px;'>
        Analyzing campaign performance and predicting user clicks.
    </p>
    <hr>
    """, 
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv('/workspaces/CTR_Prediction_ML/ad_10000records.csv')

df = load_data()
st.header("Campaign Overview")
col1, col2, col3 = st.columns(3)

total_impressions = len(df)
total_clicks = df['Clicked on Ad'].sum()
ctr = (total_clicks / total_impressions) * 100

col1.metric("Total Impressions", f"{total_impressions:,}")
col2.metric("Total Clicks", f"{total_clicks:,}")
col3.metric("Average CTR", f"{ctr:.2f}%")

st.header("Audience Insights: Age vs. Clicks")

df['Action'] = df['Clicked on Ad'].map({0: 'Ignored Ad', 1: 'Clicked Ad'})

fig = px.histogram(df, x="Age", color="Action",
                   color_discrete_map={'Ignored Ad': '#a2cffe', 'Clicked Ad': '#ffb347'},
                   barmode='stack')

fig.update_layout(
    xaxis_title="User Age",
    yaxis_title="Number of Users",
    bargap=0.05, 
    plot_bgcolor="rgba(0,0,0,0)" 
)

# Display the interactive chart
st.plotly_chart(fig, use_container_width=True)

st.header(" Bid Predictor")
st.write("Adjust the sliders below to see if the ML model recommends bidding on this user.")

user_age = st.slider("User Age", 18, 70, 30)
time_on_site = st.slider("Daily Time Spent on Site (minutes)", 30, 100, 60)

if time_on_site > 65 and user_age > 25:
    st.success("Prediction: HIGH probability of click. Action: Place Bid!")
else:
    st.error("Prediction: LOW probability of click. Action: Skip Bid.")