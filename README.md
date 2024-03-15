# Welcome to My Capstone Project!
<div style="display: flex; align-items: center; justify-content: center; text-align: center;">
  <img src="https://coursereport-s3-production.global.ssl.fastly.net/uploads/school/logo/219/original/CT_LOGO_NEW.jpg" width="100" style="margin-right: 10px;">
</div>

## Predicting MLB Players' Salaries Using Machine Learning
Welcome to my Coding Temple Data Analytics Capstone! In this project, I will be using concepts taught to us across the course of the bootcamp to predict the 2023 salaries of the players on the Atlanta Braves. In this README, I will be covering my methodology and the steps I took to create and deliver this project. The target variable is the average yearly salary of the players.

The datasets I used were sourced from Baseball-Reference and Spotrac, which you can access below:
- https://www.baseball-reference.com/teams/ATL/2023.shtml
- https://www.spotrac.com/mlb/atlanta-braves/payroll/

You can also find a data dictionary defining each column of the dataset at the end of this README.

### What Is In This Repository?
- A folder called **datasets** which contains the datasets I downloaded and created.
- A folder called **images** which has two photos I used for my Streamlit app.
- A Jupyter Notebook .ipynb file in which I did most of my work.
- A Python .py file that contains the code for building my Streamlit app. This is where most of my .ipynb file eventually went.
- This README, outlining my methodology for my project.
- A PDF file of a slide deck I created for my presentation of the project in class.
  
**STEP 1: Importing Libraries and Reading in CSVs**
> The necessary Python libraries needed for this project are pandas, seaborn, matplotlib.pyplot, and numpy. I also used Plotly, but that is not included here in the analysis, it is imported on the VSCode workbook connected to my Streamlit app, which I will mention later. I also used certain features from scikit-learn for my machine learning models: LinearRegression,RandomForestRegressor, cross_val_score, r2_score, mean_squared_error, and KNeighborsRegressor. I then read in the CSVs for batting, pitching, and fielding statistics downloaded from Baseball-Reference, as well as reading in a dataset containing contract and payroll information I manually created, using the data provided from Spotrac. I did not have an option to download it. I then merged the datasets together to make one large dataset I could use for the rest of the project.

**STEP 2: Cleaning the Dataset and EDA**
> First, I dropped a bunch of columns that had barely any values and were basically useless. I also sorted the players by position to organize batting and pitching statistics separately while still being in the same dataset. I then renamed a bunch of columns that were similarly named, as merging the datasets had some columns with different data, but the same name. I also needed to fill in null values, as pitchers do not have hitting statistics and hitters do not have pitching statistics (for the most part). I think this structure may have affected my results, which would be something I would be interested in investigating during my free time. I also needed to get dummies of categorical variables to turn them into numbers, as predictive ML models cannot use categorical variables. As for my EDA, I used seaborn's pairplot method to take a look at some graphs and see if I could identify any trends in the data that would be useful to my model. I also used a heatmap to identify correlation between my target variable (average yearly salary) and the rest of the variables.

**STEP 3: Building My Models**
> Features that were to be included in my model were determined by the pairplot and heatmap I created. I then created a baseline model using the mean of average player salary. Once the baseline model was done, I built Linear Regression, Random Forest, and KNN models. The measurement for model sucess I used was the root mean squared error(RMSE).

**Conclusion**
> Every model had a better RMSE score than the baseline model by a significant margin, but it still wasn't super accurate. You can view my Streamlit to find a more in-depth analysis.

### Data Dictionary
**HITTING STATISTICS**

- G: Games Played
- PA: Plate Appearances
- AB: At-Bats
- R: Runs
- H: Hits
- 2B: Doubles
- 3B: Triples
- HR: Home Runs
- RBI: Runs Batted In
- SB: Stolen Bases
- CS: Caught Stealing
- BB: Walks
- SO: Strikeouts
- BA: Batting Average
- OBP: On-Base Percentage
- SLG: Slugging Percentage
- OPS: On-Base + Slugging Percentage
- OPS+: OPS that has been adjusted to the player's ballpark dimensions
- TB: Total Bases
- GDP: Double Plays Grounded Into
- HBP: Hit By Pitch
- SH: Sacrifice Hits
- SF: Sacrifice Flies
- IBB: Intentional Walks

**PITCHING STATISTICS**
- W: Wins
- L: Losses
- W-L%: Win Loss Percentage
- ERA: Earned Run Average
- G: Games Played
- GS: Games Started
- GF: Games Finished
- CG: Complete Games
- SHO: Shutouts (no runs allowed during the game)
- SV: Saves
- IP: Innings Pitched
- H: Hits Allowed
- R: Runs Allowed
- ER: Earned Runs Allowed
- HR: Home Runs Allowed
- BB: Walks Allowed
- IBB: Intentional Walks Allowed
- SO: Strikeouts
- HBP: Hit By Pitches
- BK: Balks
- WP: Wild Pitches
- BF: Batters Faced
- ERA+: ERA adjusted to player's ballpark dimensions
- FIP: Fielding independent pitching (measures strikeouts, walks, HBP, and home runs, things only the pitcher can control)
- WHIP: Walks and Hits Per Inning
- H9: Hits Allowed Per 9 Innings
- HR9: Home Runs Allowed Per 9 Innings
- BB9: Walks Allowed Per 9 Innings
- SO/9: Strikeouts Per 9 Innings
- SO/W: Strikeout to Walk Ratio

**FIELDING STATISTICS**
- G: Games Played or Pitches
- GS: Games Started
- CG: Complete Game
- Inn: Total Innings Played
- Ch: Defensive Chances
- PO: Putouts
- A: Assists
- E: Errors
- DP: Double Plays
- Fld%: Fielding Percentage
- Rtot: The number of runs above or below average the player is worth based on the number of plays made
- Rtot/yr: Rtot per 1200 innings
- Rdrs: The number of runs above or below average the player is worth based on the number of plays made
- Rdrs/yr: Rdrs per 1200 innings
- Rgood: The number of runs above or below average the player is worth based on the number of plays made where they made an exceptional contribution or they obviously misplayed it (I hate subjective stats like these, personally. I dropped it for good reason.)
- RF/9: Range factor per 9 innings ( 9 * (Putouts+Assists) / Innings Played)
- RF/G: Range factor per game (Putouts+Assists)/Games Played
- PB: Passed Balls
- WP: Wild Pitches
- SB: Stolen Bases
- CS: Caught Stealing
- CS%: Caught Stealing Percentage
- lgCS%: League Expected CS / (SB + CS)
- PO: Pickoffs
- Pos Summary: Positions Played

**CONTRACT/GENERAL STATISTICS**
- Player: Name of player
- Age: Player's age on midnight of June 30th of that year
- Position: Position player plays
- Total Salary: Total amount of contract in USD
- Contract Length: Length of contract in years
- Average Yearly Salary: Total Salary / Contract Length
  
