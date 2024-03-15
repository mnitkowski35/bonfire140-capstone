import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Reading in data 
df = pd.read_csv('datasets/newdataframe.csv')

# Page Title and Icon
st.set_page_config("Predicting MLB Salaries", page_icon = ":baseball:")
# Sidebar
st.sidebar.header("Navigate to other pages here.")
page = st.sidebar.selectbox("Select A Page",['Introduction','Data Overview','Exploratory Data Analysis','Predictive Modeling','Conclusion'])
st.sidebar.divider()
st.sidebar.write("I recommend reading the pages in the order in which they appear on the dropdown menu in order to better understand the context of the page and data analysis as a whole, but feel free to explore the data however you like!")
st.sidebar.divider()
st.sidebar.write("This website was completed as a capstone project for CodingTemple's data analytics bootcamp. A special thanks to Katie Sylvia and the team at CodingTemple for their wonderful guidance and instruction in building this application.")

# HOMEPAGE
if page == "Introduction":
    st.title("Analyzing MLB Salaries: Predicting the 2023 Salaries of Atlanta Braves Baseball Players")
    st.subheader("Presented by Matthew Nitkowski")
    st.write("Hello! My name is Matthew Nitkowski, and I am an aspiring data analytics student. Baseball has always been a passion of mine and my family- without it, I likely wouldn't even be here. My father played Major League baseball growing up, so I've been around the game since I was a child and even played baseball myself. One thing I love in particular about the game today is the amount of data that can be extracted and analyzed, and what we can do with these numbers to innovate how we can change the game and see the game through an entirely new lens.")
    st.image("images/2011 CJ Brooke Matt Turner.jpeg")
    st.caption("My father, CJ (center), my sister, Brooke (right), and myself (left) at the 2011 Atlanta Braves Alumni Weekend")
    st.write("The goal of this project is to build a predictive regression model that can accurately predict the individual salaries of the 2023 Atlanta Braves based on the statistics they provided that year. Throughout this project, I will be demonstrating programming and presentation concepts I've learned throughout my time at CodingTemple.")
    st.image("images/2011 Braves Game Fam.jpg")
    st.caption("My family at a 2011 Braves game, including my mother, Megan, and my brother, Luke")
# DATA OVERVIEW
if page == "Data Overview":
    st.header("Data Overview: What Do We Want to Accomplish?")
    st.write("The goal of this project is to be able to accurately predict the individual salaries of the 2023 Atlanta Braves with machine learning. Since many players' contracts span multiple years, the target variable is the average yearly salary of each individual player.")
    st.write("In most of our CodingTemple projects, we were given a single dataset to clean and analyze. However, given the variability and complexity of this undertaking, it was necessary to retrieve multiple datasets from multiple sources and combine them.")
    st.link_button("The batting, fielding, and pitching statistics were sourced from a website called Baseball-Reference. Click here to view the original source.","https://www.baseball-reference.com/teams/ATL/2023.shtml")
    st.link_button("Contract and payroll information were sourced from a website called Spotrac. Click here to learn more from the original source.","https://www.spotrac.com/mlb/atlanta-braves/payroll/")
    st.divider()
    st.header("Taking A Look at the Dataset :bar_chart:")
    st.write("Click on the boxes below to view some information about the dataset!")
    if st.checkbox("Dataframe and Shape"):
        st.write(f"Below is the dataframe of the final dataset I created. There are {df.shape[0]} players being analyzed and {df.shape[1]} statistics and categories in which they are being analyzed.")
        st.write("You can interact with the dataframe below! Click on any column to view statistics in ascending or descending order.")
        st.write("The data dictionary for this dataset is stored in the Github repository for my capstone. Click here to view the repository and what each stat means!")
        st.dataframe(df)
    if st.checkbox("Column List and Arbitration"):
        st.code(f"Columns: {df.columns.tolist()}")
        st.write("Some of the columns below may look a little odd:")
        ls = ['Status_Veteran','Status_Arbitration Year 2','Status_Arbitration Year 3','Status_Pre-Arbitration']
        s = ''
        for category in ls:
            s += '- ' + category + "\n"
        st.markdown(s)
        st.write("This is because Status is a categorical variable and not numerical. In order to build a predictive model, all of our variables that will be used in the model need to be numerical. I needed to convert the category status to a numerical value. A 1 means that this is the player's status, and a zero means it is not the player's status.")
        st.subheader("What is arbitration?")
        st.markdown("In baseball, **arbitration** is a negotation method that when a player and a team cannot agree on a salary to be paid, it'll be sent to a third party arbitrator to decide which side's valuation is more accurate. Whoever the arbitrator deems is more accurate is the salary the player will be paid.")
        st.write("A player's arbitration years are in their 3rd to 6th years of service time, which is the amount of time that has passed since the player's first day on an MLB roster. If a player signs a contract with the team, they can avoid arbitration. Some players' contracts can last until after their arbitration eligibility expires, so for the sake of this project, if that is the case, they are given veteran status.")
        st.write("An example of this in this dataset is Braves' outfielder Michael Harris II. Harris was only 22 years old in 2023, but agreed to an 8-year, $72 million contract that expires in 2032. Because of this, even though he does not have 6+ years of MLB service time and is technically not a veteran, he is classified as a Veteran to simplify this project.")
        st.write("Players who have less than 3 years of service time are usually paid around the league minimum of $720,000 unless the team and player agree to a contract worth more than that. In Harris' case, he was offered a long-term deal that elevated him from Pre-Arbitration to Veteran status.")

# EDA
if page == 'Exploratory Data Analysis':
    st.title(":mag_right: Time to Analyze the Data! :mag:")
    st.write("On this page, we will be analyzing the individual relationships between any variables of your choosing with three charts- histograms, scatterplots, and boxplots.")
    cols = df.select_dtypes(include = 'number').columns.tolist()
    st.markdown("**Keep in mind that some charts may look a little odd as some categorical variables have been converted to numerical variables.**")
    eda_type = st.multiselect("Choose a visualization you are interested in exploring. You can select more than one at a time!",['Histograms','Box Plots','Scatterplots'])
    if 'Histograms' in eda_type:
        st.subheader("Histograms")
        st.markdown("A **histogram** is a graph that shows the frequency of numerical data.")
        histogram_col = st.selectbox("Choose a column for your histogram!",cols,index = None)
        if histogram_col:
            st.plotly_chart(px.histogram(df, x = histogram_col, title = f"Distribution of {histogram_col.title()}"))
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots")
        st.markdown("A **box plot**, also known as a box and whisker diagram, is a graph summarizing a set of data. The shape of the box plot shows how data is distributed and also includes outliers.")
        boxplot_col = st.selectbox("Select a column for your box plot!", cols, index = None)
        if boxplot_col:
            st.plotly_chart(px.box(df,x = boxplot_col, title = f"Distribution of {boxplot_col.title()}"))
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots")
        st.markdown("A **scatterplot** provides a visual means to test the strength of a relationship between two variables. It plots the point where the specified x and y value for each indvidual passenger meets, and creates a **line of best fit** to predict values.")
        selected_col_x = st.selectbox("Select x-axis variables:", cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variables:", cols, index = None)
        if selected_col_x and selected_col_y:
            st.plotly_chart(px.scatter(df, x= selected_col_x, y = selected_col_y, title = f"Relationship Between {selected_col_x.title()} and {selected_col_y.title()} "))
# Predictive Modeling
if page == "Predictive Modeling":
    st.header("Time to Model! :gear:")
    st.markdown("**Predictive Modeling** is the concept of using models to predict what our target variable is going to be based on data we are given. In this application, we are attempting to predict what statistics influence how players get paid.")
    st.write("There are many different type of machine learning predictive models we can use. On this application, we will be using three models.")
    st.subheader("What Will We Use to Measure Our Model?")
    st.markdown("The measure for our model will be the **root mean squared error(RMSE)**, which is measures the average difference between values predicted by a model and the actual values. It provides an estimation of how well the model is able to predict yearly average salaries.")
    st.image("https://arize.com/wp-content/uploads/2023/08/rmse-example-regression.jpg")
    st.caption("A visualization of root mean squared error. The label 'diff' is the RMSE.")
    st.divider()
    st.subheader("How Do We Know if Our Model is Any Good?")
    st.markdown("In order to see if our model is any good, we need to create a **baseline model**, which is a simple predictive model that predicts values based on usually one statistic. For our baseline model, we will be using the mean.")
    st.markdown("When using the mean, our RMSE score is $6,336,312. This means that the average predicted value was off from the actual value by over six **MILLION** dollars! We can do better than that!")
    st.write("These are the statistics that we will be using to predict player salaries:")
    l = ['Contract Length','HR: Home Runs','Status: Pre-Arbitration','Status: Veteran','ERA: Earned Run Average','FIP: Fielding Independent Pitching,','HR9: Home Runs Allowed Per 9 Innings','H9: Hits Allowed Per 9 Innings','WHIP: Walks and Hits Per Innings Pitched','BB: Walks','SO_x: The amount of times batters struck out']
    s = ''
    for category in l:
        s += '- ' + category + "\n"
    st.markdown(s)
    # Setting up our model
    features = ['Contract Length','HR','Status_Pre-Arbitration','Status_Veteran','ERA','FIP','HR9','H9','WHIP','BB','SO_x']
    X = df[features]
    y = df['Average Yearly Salary']
    # Model Selection
    st.subheader("Select a model below! You may only choose one model at a time.")
    model_option = st.selectbox("Choose a model!",['Linear Regression','Random Forest','KNN'], index = None)
    if model_option:
        if model_option == 'Linear Regression':
          st.write("Linear regression predicts the value of a variable based on the value of another variable.")
          model = LinearRegression()
        elif model_option == 'Random Forest':
            st.write("Random Forest is a commonly-used machine learning algorithm that combines the output of multiple decision trees to reach a single result.")
            model = RandomForestRegressor()
        elif model_option == 'KNN':
            st.write("KNN, also known as K-Nearest-Neighbors, is a model that predicts the value of a variable based on its nearest neighbors, which is the value k.")
            st.write("K needs to be an odd number so we do not end up with a tie if there is a variation in nearest neighbors.In KNN tests, the default value for k is 5.")
            k_value = st.slider("Select the number of k:",min_value = 1, max_value = 29, step = 2, value= 5)
            
            model = KNeighborsRegressor(n_neighbors = k_value, n_jobs = -1)
        if st.button("Let's check out our results! It may take a bit to load."):
            model.fit(X,y)
            predicted_values = model.predict(X)
            rmse = round(np.sqrt(mean_squared_error(y, predicted_values)))
            st.subheader(f"{model} Evaluation:")
            st.write(f"Our RMSE for {model} is {rmse}.")

# Conclusion
if page == 'Conclusion':
    st.header("Final Machine Learning Results!")
    st.subheader("We finally made it to our conclusion! But what did we learn from our predictive machine learning models?")
    st.write("Remember, we want our predictive models we built to achieve a better RMSE score than $6,336,312.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Linear Regression Results")
        st.subheader("RMSE Score: $3.8 million")
        st.write("With a linear regression model, we are already seeing a massive improvement from our baseline model, but Linear Regression is still a pretty weak model. 3.8 million dollars is a wide gap.")
    with col2:
        st.header("Random Forest Results")
        st.subheader("RMSE Score: $1.7-2 million")
        st.write("The Random Forest model produces varying results depending on the decision tree it follows, but the range is around 1.7 to 2 million dollars. This is a much better model, but still a pretty high RMSE.")
    with col3:
        st.header("KNN(3 Neighbors) Results")
        st.subheader("RMSE Score: $3.4 million")
        st.write("KNN results also vary depending on the number of neighbors. Generally, the lower the amount of neighbors, the more accurate the results. Our best result was with 3 neighbors resulting in an RMSE score of $3.4 million dollars.")
    st.subheader("What Did We Learn?")
    st.write("Our best model was the Random Forest Regressor with a RMSE score of $1.7 million dollars. While this is a 4-5 million dollar improvement from our baseline model, there still is a lot of room for error.")
    st.write("There could be a lot of reasons for this. One explanation I have for this is that the dataset is small, containing only 37 players. Machine learning models are able to make more accurate predictions when fed large amounts of data. If we replicated this project with every active MLB player in 2023, it is likely we would see more accurate results.")
    st.write("It is difficult to predict baseball salaries. Not everything can be explained by numbers. Other factors, such as a player's marketability, may factor in why players are paid more. Generally, fans want to see good players, but if a player very marketable, they can sacrifice a little bit of skill and still get paid more compared to less marketable players that have better stats. But that's just a theory.")
    st.write("Another theory I have regarding the high RMSE scores is regarding the data itself. Combining all of the players and all of the statistics together meant that a lot of null values were created. Most pitchers don't have hitting statistics and most hitters don't have pitching statistics. Because of this, it may have cause some correlation issues and futhermore impacted how the model interpreted how players got paid.")
    st.write("Regardless about the varying results, this project was incredibly fun to make. Numbers and statistics have really changed the game over the years and I love to be a part of it. I hope you enjoyed as well and learned a little bit about the game of baseball!")



