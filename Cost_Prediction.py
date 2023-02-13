import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
# import math
# from predict_cost import predict
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from streamlit_option_menu import option_menu
import base64
from PIL import Image

#load image
lr = Image.open('img/mulienarregbefore.png')
ridge   = Image.open('img/ridgebefore.png')
lasso   = Image.open('img/lassobefore.png')
dt_before  = Image.open('img/dtbefore.png')
rf_before = Image.open('img/rfbefore.png')
lgbm_before = Image.open('img/lgbmbefore.png')
dt1 = Image.open('img/dt1.png')
dt2 = Image.open('img/dt2.png')
rf1 = Image.open('img/rf1.png')
rf2 = Image.open('img/rf2.png')
ransac1 = Image.open('img/ransac1.png')
ransac2 = Image.open('img/ransac2.png')
lgbm1 = Image.open('img/lgbm1.png')
lgbm2 = Image.open('img/lgbm2.png')
xgb1 = Image.open('img/xgb1.png')
xgb2 = Image.open('img/xgb2.png')

yodi = Image.open('img/Yodi.jpg')
 

#create function for background url
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    ''' 
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-vector/hand-painted-watercolor-pastel-sky-background_23-2148902771.jpg?w=2000");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#call function background
set_bg_hack_url()




# Load the pre-trained model from the .pkl file
# with open("BPJS_CostPrediction.pkl", "rb") as f:
#     model = pickle.load(f)
model = joblib.load("BPJS_CostPrediction_xgb.pkl")



# Create a function to make predictions with the model
def predict(data):
    # inputs = data
    prediction = model.predict(data)
    prediction = prediction.astype(int)
    return prediction

#page for home
def home():
    st.title("BPJS Hackathon Cost Prediction")
    st.write("In this project we will predict Cost for BPJS Hackathon dataset from Kaggle. The features has been encoded by Author (BPJS)")
    st.write("So, first of all we will Explore the data to find outlier or missing value. The dataset has 36 columns and more 50k rows.")
    st.write("Before we going through, we will try with baseline model to know how much performance that we got, and then we will deep dive with Cleansing and Tunning Hyperparameter")
    st.write("The following dataset can be accessed [here](https://www.kaggle.com/datasets/bagusbpg/bpjs-kesehatan-hackathon-2021-cost-prediction)")
    st.write("Full code can be accessed here :sunglasses:  [source](https://github.com/yodialfa/BPJSCostPrediction-Hackathon)")
    

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("https://wartapemeriksa.bpk.go.id/wp-content/uploads/2021/01/bpjs-kesehatan.jpg")

    with col3:
        st.write(' ')

    st.header("Data Understanding")
    st.write("In this dataset we have 36 columns, and 57.971, but I don't know description per each columns because the author was encoded the features."
                "some data we can know the description, like tglpelayanan, peserta, case, and unit_cost."
                "tglpelayanan is date of serve, peserta is encoded of member, case total case of serve, unit cost is the target"
            )
    st.header("Data Cleansing")
    st.write("In this section I drop some features like rowid which number of row, tglpelayanan, peserta and case."
             "and when I check missing value I didn't found missing value."
            )
    st.header("Baseline Modelling")
    st.write("I split the data to train and test with ratio 80/20, then modelling with baseline model (Linear Regression, Ridge and Lasso."
             "And I have r2 score is 0.89 and rmse is 607.313 (Multiple Linear Regression), with Ridge is 0.89 and rmse is 607.355 (Ridge)."
             "And with Lasso r2 score is 0.89 too, and RMSE is 607.315.")
    st.write("I try Decission Tree, Random Forest and LightGBM without hyper parameter tunning, And I got R2 Score and RMSE (0.97 , 272.031 (DT))"
             "R2 Score 0.98 RMSE = 229.257 (Random Forest), R2 Score = 0.976 Rmse = 283.011 (Light GBM). It's good than Linear Regression, Ridge and Lasso" 
            ) 
    st.subheader("Multiple Linear Regression Before Deep Cleansing And Tunning Hyperparmeter")
    st.image(lr, caption='Multiple Linear Regression')
    st.subheader("Ridge Regression Before Deep Cleansing And Tunning Hyperparmeter")
    st.image(ridge, caption='Ridge Regression')
    st.subheader("Lasso Regression Before Deep Cleansing And Tunning Hyperparmeter")
    st.image(lasso, caption='Lasso Regression')
    st.subheader("Decission Tree Regression Before Deep Cleansing And Tunning Hyperparmeter")
    st.image(dt_before, caption='Decission Tree Regression')
    st.subheader("Random Forest Regression Before Deep Cleansing And Tunning Hyperparmeter")
    st.image(rf_before, caption='Random Forest Regression')
    st.subheader("LGBM Regression Before Deep Cleansing And Tunning Hyperparmeter")
    st.image(lgbm_before, caption='LGBM Regression')
    st.header("Deep Cleansing")       
    st.write("After using baseline for our data. I try to cleaning the data again. First of all I drop 'ds' because has 1 value, and I drop 'peserta too'"
             "Because member id isn't using for modelling." 
            ) 
    st.write("And I see on target (unit_cost) didn't normal. So I try to doing remove outlier for the next step."
             "But after we doing removing outlier, the target still didn't normal. So I doing Feature Engineering using Log Transformation. And After that the data" 
             "is normal, but bimodal(two peak). So I try to going the next step because some ML has immune that bimodal data"
            ) 
    st.header("Modelling and Development")
    st.write("After deep cleansing, I try to fit using some model and without tunning hyperparameter. And I got value like this :")

    st.subheader("Decission Tree Regressor")
    st.write("Decission Tree : R2 Score = 0.99 RMSE = 0.057609")
    st.image(dt1, caption='Decission Tree Regressor Before Tunning')

    st.subheader("Random Forest Regressor")
    st.write("Random Forest : R2 Score = 0.99 RMSE = 0.057409")
    st.image(rf1, caption='Random Forest Regressor Before Tunning')

    st.subheader("RANSAC Regressor")
    st.write("Ransac Regressor : R2 Score = 0.976 RMSE = 0.0912")
    st.image(ransac1, caption='RANSAC Regressor Before Tunning')

    st.subheader("Light GBM")
    st.write("Light GBM : R2 Score =0.986 RMSE = 0.068224")
    st.image(lgbm1, caption='LGBM Before Tunning')

    st.subheader("XGB Regressor")
    st.write("XGB Regressor : R2 Score = 0.989 RMSE 0.06007")
    st.image(xgb1, caption='XGB Regressor Before Tunning')
    
    st.subheader("Improvements")
    st.write("And I try to improve that model with Hyperparameter Tunning. So I have score like this")

    st.subheader("Decission Tree Regressor After Tunning Hyperparameter")
    st.write("Decission Tree : R2 Score = 0.973 RMSE = 0.0966")
    st.image(dt2, caption='Decission Tree Regressor After Tunning')

    st.subheader("Random Forest Regressor After Tunning Hyperparameter")
    st.write("Random Forest : R2 Score = 0.988 RMSE = 0.06575")
    st.image(rf2, caption='Random Forest Regressor After Tunning')

    st.subheader("RANSAC Regressor After Tunning Hyperparameter")
    st.write("Ransac Regressor : R2 Score = 0.976 RMSE = 0.09106")
    st.image(ransac2, caption='RANSAC Regressor After Tunning')

    st.subheader("Light GBM After Tunning Hyperparameter")
    st.write("Light GBM : R2 Score =0.989 RMSE = 0.06044")
    st.image(lgbm2, caption='LGBM After Tunning')

    st.subheader("XGB Regressor After Tunning Hyperparameter")
    st.write("XGB Regressor : R2 Score = 0.99 RMSE 0.05686")
    st.image(xgb2, caption='XGB Regressor After Tunning')

    st.write("We have some result from the model, before hyperparameter tunning DT and RF we got slowest rmse, but after tunning the rmse is high.so we can assumed that the model is overfitting.  Ransac Regressor after tunning hyperparameter"
             "show improvements, but the RMSE still high. ")
    st.write("LGBM and XGB Regressor seem has improvement after Tunning Hyperparmeter. But between that model, XGB is the slowest RMSE than LGBM"
             "And I'll choose XGB for the model because the model is the best than other."
            )


#page for project
def project():
    # Create a Streamlit app
    st.title("BPJS Hackathon")
    html_temp = """
    <div style="background-color:#e9a296;padding:10px">
    <h2 style="color:white;text-align:center;">Cost Prediction </h2>
    </div>

    """
    page_bg_img = f"""
    <style>
    body {{
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img,unsafe_allow_html=True)
    # Get input from the user
    st.markdown(html_temp,unsafe_allow_html=True)
    kddati2 = st.number_input("Enter kkdati2:", min_value=0, step=1)
    tkp = st.selectbox("Select TKP:", (30, 40))
    a = st.selectbox("Select a:", (0, 1, 2))
    b = st.selectbox("Select b:", (0, 1, 2, 3, 4, 5, 6, 7, 8))
    c = st.selectbox("Select c:", (0, 1, 2, 3, 4, 5))
    cb = st.selectbox("Select cb:", (0, 1))
    d = st.selectbox("Select d:", (0, 1, 2, 3, 4, 5))
    gd = st.selectbox("Select gd:", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    hd = st.selectbox("Select hd:", (0, 1, 2, 3, 4, 5, 6))
    i1 = st.selectbox("Select i1:", (0, 1))
    i2 = st.selectbox("Select i2:", (0, 1, 2))
    i3 = st.selectbox("Select i3:", (0, 1, 2, 3))
    i4 = st.selectbox("Select i4:", (0, 1, 2, ))
    kb = st.selectbox("Select kb:", (0, 1, 2))
    kc = st.selectbox("Select kc:", (0, 1))
    kg = st.selectbox("Select kg:", (0, 1, 2, 3, 4))
    ki = st.selectbox("Select ki:", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11))
    kj = st.selectbox("Select kj:", (0, 1, 2))
    kk = st.selectbox("Select kk:", (0, 1))
    kl = st.selectbox("Select kl:", (0, 1, 2, 3, 4, 5, 6))
    km = st.selectbox("Select km:", (0, 1, 2, 3, 4, 5, 6))
    ko = st.selectbox("Select ko:", (0, 1))
    kp = st.selectbox("Select kp:", (0, 1, 2))
    kt = st.selectbox("Select kt:", (0, 1))
    ku = st.selectbox("Select ku:", (0, 1))
    s = st.selectbox("Select s:", (0, 1, 2, 3, 4, 5))
    sa = st.selectbox("Select sa:", (0, 1))
    sb = st.selectbox("Select sb:", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    sc = st.number_input("Enter range 0-25:", min_value=0, step=1)
    sd = st.selectbox("Select sd:", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    # Make a prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        prediction = predict(np.array([[
            kddati2, tkp, a, b, c, cb, d, gd, hd, i1, i2, i3, i4, kb, kc, kg, ki,
            kj, kk, kl, km, ko, kp, kt, ku, s, sa, sb, sc, sd
            ]]))

        temp = prediction[0]
        # z = np.array(10**temp)
        currency = "Rp. {:,.2f}".format(temp)
        # main_currency, fractional_currency = currency.split('.')
        new_main_currency = currency.replace(',', '.')
        currency = new_main_currency
        st.write("Cost Prediction     :  ",currency)
  
        
def contact():
    # --- GENERAL SETTINGS ---
    PAGE_TITLE = "Digital CV | John Doe"
    PAGE_ICON = ":wave:"
    NAME = "Yodi Ramadhani Alfariz"
    DESCRIPTION = """
    Data Analyst, Data Scientist, Business Intelligent
    """
    EMAIL = "yodialfariz@gmail.com"
    LINKEDIN ="https://linkedin.com/in/yodialfariz"
    GITHUB ="https://github.com/yodialfa"
    TWITTER = "https://twitter.com/yodiumh"
 

    # --- HERO SECTION ---
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(yodi, width=230)

    with col2:
        st.title(NAME)
        st.write(DESCRIPTION)
        st.write("📫", EMAIL)
        st.write(LINKEDIN)
        st.write(GITHUB)
        st.write(TWITTER)



#navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home","Projects","Contact"],
    icons=["house","book","envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#E9A296"},
            },
)

#selected page
if selected == "Home":
    home()
if selected == "Projects":
    project()
if selected == "Contact":
    contact()