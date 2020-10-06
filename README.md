# NFL, MLB, NBA and Soccer scores
1. Set up a data science project structure in a new git repository in your GitHub account
2. Pick one of the game data sets depending your sports preference
                https://github.com/fivethirtyeight/nfl-elo-game
                https://github.com/fivethirtyeight/data/tree/master/mlb-elo
                https://github.com/fivethirtyeight/data/tree/master/nba-carmelo
                https://github.com/fivethirtyeight/data/tree/master/soccer-spi
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5. Build one or more regression models to determine the scores for each team using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

# Idea about feature engineering

In a soccer game, there is an extremely important feature that is easily overlooked. It is home and away information. For the soccer team, the home stadium means a more familiar environment, more affectionate fans, and more enthusiastic voices. Perhaps, this is difficult to reflect from the data, but this potential data has an important influence on the team's victory.

For the away team, the weight will be lower than the home team, which allows us to get more accurate results when predicting the score.

We hope to introduce home and away information to assist our regression model to learn more accurate regression equations to predict the team's score.
