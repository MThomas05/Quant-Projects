# File: Black-ScholesPricer.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

## Page Configuration 

st.set_page_config(page_title="Black-Scholes Option Pricer", layout ="wide", initial_sidebar_state = "expanded")

## Black Scholes model

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S # Spot price
        self.K = K # Exercise price
        self.T = T # Time to maturity
        self.r = r # Interest rate annualised
        self.sigma = sigma # Volatility

    ## Calulcation of d1 and d2 using BS Model

    def d1_d2(self):
        d1 = (np.log(self.S/self.K) + (self.r + self.sigma**2/2) * self.T) * (1/(self.sigma*np.sqrt(self.T)))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    ## Calculation of call and put values

    def optionprice(self, option_type):
        d1, d2 = self.d1_d2()
        if option_type == "Call Option":
                return norm.cdf(d1) * self.S - norm.cdf(d2) * self.K * np.exp(-self.r * self.T)
        elif option_type == "Put Option":
            return norm.cdf(-d2) * self.K * np.exp(-self.r * self.T) - norm.cdf(-d1) * self.S
        
        
    ## Define the greeks (delta, gamma, vega, theta, rho)

    def greek_values(self, option_type):
        d1, d2 = self.d1_d2()
        if option_type == "Call Option":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        if option_type == "Call Option":
            gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        else:
            gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

        if option_type == "Call Option":
            vega = self.S * np.sqrt(self.T) * norm.pdf(d1)
        else:
            vega = self.S * np.sqrt(self.T) * norm.pdf(d1)

        if option_type == "Call Option":
            theta = ( - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                    - (self.r * self.K * np.exp(-self.r / self.T) * norm.cdf(d2)))
        else:
            theta = ( - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                    + (self.r * self.K * np.exp(-self.r / self.T) * norm.cdf(-d2)))

        if option_type == "Call Option":
            rho = (self.K * self.T * np.exp(-self.r * self.T) *norm.cdf(d2)) / 100
        else:
            rho = ( self.K * self.T * np.exp( -self.r * self.T) * norm.cdf(-d2)) / -100
            

            ## Return as a dictionary

        return{"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
                    
with st.sidebar:
    st.markdown("<h3 style='text-align: center; colour: White;'>Option Type</h3>", unsafe_allow_html = True)
    option_type = st.selectbox("", ["Call Option", "Put Option"])
    st.subheader("Choose Option Parameters")
    S = st.number_input("Underlying Asset Price", value=100.0)
    K = st.number_input("Strike Price", value=100.0)
    T = st.number_input("Time To Maturity - Years", value=1.0)
    r = st.number_input("Risk-Free Interest Rate", value=0.05)
    sigma = st.number_input("Volatility", value=0.2)
    

st.title("Black-Scholes Option Pricing Model")

input_data = {"Underlying Asset Price": [S],
                "Strike Price": [K],
                "Time To Maturity - Years": [T],
                "Risk-Free Interest Rate": [r],
                "Volatility": [sigma]}

input_df = pd.DataFrame(input_data)
st.table(input_df)

col1, col2 = st.columns(2)

import plotly.graph_objects as go

def map_option_prices(variables, values, S, K, T, r, sigma, option_type):
    if variable == "Underlying Asset Price":
        price = [BlackScholesModel(val, K, T, r, sigma).optionprice(option_type) for val in values]
    if variable == "Strike Price":
        price = [BlackScholesModel(S, val, T, r, sigma).optionprice(option_type) for val in values]
    if variable == "Time To Maturity - Years":
        price = [BlackScholesModel(S, K, val, r, sigma).optionprice(option_type) for val in values]
    if variable == "Risk-Free Interest Rate":
        price = [BlackScholesModel(S, K, T, val, sigma).optionprice(option_type) for val in values]
    elif variable == "Volatility":
        price = [BlackScholesModel(S, K, T, r, val).optionprice(option_type) for val in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values, y=price, mode='lines', name=f"{option_type} Option Price"))
    fig.update_layout(title=f"{option_type} Option Price vs. {variable}", xaxis_title=variable, yaxis_title="Option Price")
    
    return fig

def map_option_greeks(variable, values, S, K, T, r, sigma, option_type):
    greeks = ["delta", "gamma", "vega", "theta", "rho"]
    fig = go.Figure()
    
    for greek in greeks:
            
        if variable == "Underlying Asset Price":
            greeks = [BlackScholesModel(val, K, T, r, sigma).greek_values(option_type)[greek] for val in values]
        if variable == "Strike Price":
            greeks = [BlackScholesModel(S, val, T, r, sigma).greek_values(option_type)[greek] for val in values]
        if variable == "Time To Maturity - Years":
            greeks = [BlackScholesModel(S, K, val, r, sigma).greek_values(option_type)[greek] for val in values]
        if variable == "Risk-Free Interest Rate":
            greeks = [BlackScholesModel(S, K, T, val, sigma).greek_values(option_type)[greek] for val in values]
        elif variable == "Volatility":
            greeks = [BlackScholesModel(S, K, T, r, val).greek_values(option_type)[greek] for val in values]

        fig.add_trace(go.Scatter(x=values, y=greeks, mode='lines', name=f"{greek} ({option_type})"))

    fig.update_layout(title=f"{option_type} Greeks vs. {variable}", xaxis_title=variable, yaxis_title="Greeks")

    return fig

model = BlackScholesModel(S, K, T, r, sigma)
option_price = model.optionprice(option_type)
greeks = model.greek_values(option_type)

with col1:
    st.markdown(f"<h2 style=' text-align: left; colour: White'>Option Price: {option_price:.2f}</h2>", unsafe_allow_html = True)

with col2:
    st.markdown(f"<h2 style=text-align: center; colour: White'>Select Variable</h2>", unsafe_allow_html = True)
    variable = st.selectbox("", ["Underlying Asset Price", "Strike Price", "Time To Maturity - Years", "Risk-Free Interest Rate", "Volatility"])
    if variable == "Underlying Asset Price":
        values = np.linspace(50, 200, 100)
    if variable == "Strike Price":
        values = np.linspace(50, 200, 100)
    if variable == "Time To Maturity - Years":
        values = np.linspace(0.1, 4, 100)
    if variable == "Risk-Free Interest Rate":
        values = np.linspace(0.0, 1, 100)
    elif variable == "Volatility":
        values = np.linspace(0.1, 5, 100)

with col1:
    st.markdown(f"<h3 style='text-align: left; colour: White'> Option Greeks </h2>", unsafe_allow_html = True)
    st.markdown(f" Delta {greeks["delta"]:.2f}", unsafe_allow_html = True)
    st.markdown(f" Gamma {greeks["gamma"]:.2f}", unsafe_allow_html = True)
    st.markdown(f" Vega {greeks["vega"]:.2f}", unsafe_allow_html = True)
    st.markdown(f" Theta {greeks["theta"]:.2f}", unsafe_allow_html = True)
    st.markdown(f" Rho {greeks["rho"]:.2f}", unsafe_allow_html = True)


with col2:
    fig_price = map_option_prices(variable, values, S, K, T, r, sigma, option_type)
    st.plotly_chart(fig_price)

with col1:
    fig_greeks = map_option_greeks(variable, values, S, K, T, r, sigma, option_type)
    st.plotly_chart(fig_greeks)
    
        
    
    
    

    

    
    
    
    
    



    
    
