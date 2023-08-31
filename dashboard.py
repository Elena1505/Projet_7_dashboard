
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():

    # Shap plot
    response = requests.post("https://creditapi-joqlneigka-uc.a.run.app/shap_plot2", json={"client_id":client_id})
    data = response.json()
    content_html = data['key']
    st.components.v1.html(content_html)
    st.write("shapley POSITIVE contribue à augmenter la probabilté de prediction du risque de défaut de crédit")
    st.write("shapley NEGATIVE permet à réduire la probabilité de prédiction du risque défaut de crédit")

if __name__ == '__main__':
    main()