#imports the modules that I will need
import pandas as pd
import numpy as np
import math
import time


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt


#(csv 1)this dataset is the main one it provides data on shots and goals and what type of shot was used
df_game_plays = pd.read_csv('data/game_plays.csv')
#(csv 2)this dataset has data on which players were involved in each play in the main dataset 
df_game_plays_players = pd.read_csv('data/game_plays_players.csv')
#(csv 3)this dataset has stats on goalies
df_game_goalie_stats = pd.read_csv('data/game_goalie_stats.csv')
#(csv 4)this dataset has stats on skaters
df_game_skater_stats = pd.read_csv('data/game_skater_stats.csv')
#(csv 5)this dataset let me match player ids with player names 
df_player_info = pd.read_csv('data/player_info.csv')


#organizes dataset so it only includes relivent columns 
df_skaters = df_game_plays_players[['player_id']][(df_game_plays_players.playerType=='Shooter') | (df_game_plays_players.playerType=='Scorer')]
df_goalies = df_game_plays_players[['player_id']][(df_game_plays_players.playerType=='Goalie')]
df = df_game_plays[['event', 'secondaryType', 'st_x', 'st_y']][(df_game_plays.event=='Goal') | (df_game_plays.event=='Shot') | (df_game_plays.event=='Missed Shot') | (df_game_plays.event=='Blocked Shot')] 

#removes spaces and dashes from the dataset since they create errors
df.secondaryType = df.secondaryType.str.replace(" ", "")
df.secondaryType = df.secondaryType.str.replace("-", "")

df_test = df_game_plays[['event', 'secondaryType', 'st_x', 'st_y']][(df_game_plays.secondaryType=='WrapAround')]
print(df_test)

df = df.reset_index()
del df['index']

df_skaters = df_skaters.reset_index()
del df_skaters['index']

df_goalies = df_goalies.reset_index()
del df_goalies['index']
2

df_skaters.rename(columns = {'player_id':'skater_id'}, inplace = True)
df_goalies.rename(columns = {'player_id':'goalie_id'}, inplace = True)

#adds skater_id from df_skaters and goalie_id from df_goalies to the main dataframe df
df = df.join(df_skaters['skater_id'])
df = df.join(df_goalies['goalie_id'])

#creates a new column for each of the shot types and put a 1 under the column that corresponds to the shot used
df['WristShot'] = np.where(df.secondaryType=='WristShot', 1,0 )
df['SnapShot'] = np.where(df.secondaryType=='SnapShot', 1,0 )
df['SlapShot'] = np.where(df.secondaryType=='SlapShot', 1,0 )
df['TipIn'] = np.where(df.secondaryType=='TipIn', 1,0 )
df['WrapAround'] = np.where(df.secondaryType=='WrapAround', 1,0 )
df['Backhand'] = np.where(df.secondaryType=='Backhand', 1,0 )

#removes the secondaryType column since it is no longer needed to tell us the shot type
df.drop(columns='secondaryType', inplace=True)

#finds distance of shots/goals based on the x and y of the shot
def dist(df): 
    middle_goal_x = 86 
    middle_goal_y = 0
    return math.sqrt((middle_goal_x - df.st_x)**2 + (middle_goal_y - df.st_y)**2)

#finds angle of shots/goals based on the x and y of the shot
def angle(df):     
    middle_goal_x = 86 
    middle_goal_y = 0
    adjacent = (middle_goal_y - df.st_y)

    if adjacent == 0:
        return 0
    else:
        return math.fabs(math.atan((middle_goal_x - df.st_x) / adjacent))

#creates new column named distance and adds the distance for each shot to df
df['distance'] = df.apply(dist, axis=1)
#creates new column named angle and adds the angle for each shot to df
df['angle'] = df.apply(angle, axis=1)


#changes the data so that it only includes relivent columns
df_game_goalie_stats = df_game_goalie_stats.groupby('player_id').agg({'savePercentage':'mean'}).reset_index()
df_game_goalie_stats.rename(columns={'player_id':'goalie_id'}, inplace=True)
df_game_skater_stats = df_game_skater_stats.groupby('player_id').agg({'goals':'sum', 'shots':'sum', 'assists':'sum', 'timeOnIce':'sum'}).reset_index()
df_game_skater_stats.rename(columns={'player_id':'skater_id'}, inplace=True)
df = df.merge(df_game_skater_stats)
df = df.merge(df_game_goalie_stats)

# fills empty savePercentage cells with the median of all of the save percentages
df.savePercentage.fillna(df.savePercentage.median(), inplace=True)

#creates new column named fullName which combines the existing columns firstName and lastName
df_player_info['fullName'] = df_player_info[['firstName', 'lastName']].apply(lambda x: ' '.join(x), axis=1)
#removes columns firstName and lastName since they are no longer needed
df_player_info.drop(columns='firstName', inplace = True)
df_player_info.drop(columns='lastName', inplace = True)

#creates a new column in df named goal and if the row is a goal then it gives it a 1 and if its not it gives it a 0
df['goal'] = np.where(df.event=='Goal', 1, 0)
#removes collumns that are not needed
df.drop(columns='event', inplace=True)
df.drop(columns='skater_id', inplace = True)
df.drop(columns='goalie_id', inplace = True)
df.dropna(inplace=True)
#makes the number of goals and no goals in the data even to help give a more accurate model score
def balance_y(df, target):
        
    size = min(df[df[target]==0].shape[0], df[df[target]==1].shape[0])
    
    goals = df[df[target]==1].sample(size, replace=False, random_state=10)
    no_goals = df[df[target]==0].sample(size, replace=False, random_state=10)
    
    return pd.concat([goals, no_goals])

#applys the balancing function to the dataframe
df = balance_y(df, target = 'goal')

X = df.drop('goal', axis=1)
y = df['goal']

#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create model
lr_model = LogisticRegression()

#fit model
lr_model.fit(X_train,y_train)

predicted = (lr_model.predict_proba(X_test.values)[:,1]>=0.575).astype(int)
probs = lr_model.predict_proba(X_test.values)
print("Score of the model is", lr_model.score(X_test, y_test))

#confusion matrix is much more accurate for showing accuracy 
'''
TP = Predicted goal where there was  a goal
TN = Predicted no goal where there was no goal
FP = Predicted goal where there was no goal
FN = Predicted no goal where here was a goal

                 Actual Value
                    -------
                    |TP|FP|
    Predicted Value -------
                    |FN|TN|
                    -------
'''

cmatrix = confusion_matrix(y_test.values, predicted)
print(cmatrix)
#f1 score evaluates the accuracy by combinding the precision and recall(positives predicted as positives:total number of positives) of a model
f1 = f1_score(y_test.values, predicted)
print(f1)
print(probs)
print(predicted)
#empty array that a prediction will be made off of based on the input data that will be added to this array
new_data = [[]]
new_cords_x = []
new_cords_y = []
running = 0
#loops the process of getting input and predicting 
'''
I decided that it would be better to loop the whole input process
so that on the day of the exbition, I wouldn't have to run it each time.
'''
while running == 0:
    time.sleep(5) 
    new_data[0].clear()
        
    #functinon that adds the x and y cords to the empty array from mouse click
    def mouse_event(event):
        new_st_x = event.xdata
        new_st_y = event.ydata
        
        #adds the x and y of each click to the lists new_cords_x and new_cords_y
        new_cords_x.append(new_st_x)
        new_cords_y.append(new_st_y)

    #displays an image of a hockey rink 
    img = plt.imread('images/hockeyrink.jpg')
    fig, ax = plt.subplots()
    plt.title("Click where you want the shot to be taken from. \nWhen done close this window.")
    #sets the size and cords of the hockey rink to match the size and cords used in the dataframe
    img = ax.imshow(img, extent=[-104, 104, -47, 47])
    #makes the hockey rink clickable and connects it to the mouse_event function that was made earlier
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    plt.axis("off")
    plt.plot()

    plt.show()

    #adds the last value of both lists, the most recent value, to the empty array
    new_data[0].append(new_cords_x[-1])
    new_data[0].append(new_cords_y[-1])
    
    #variables used to break while loops
    askplayer = 0
    askgoalie = 0
    askshot = 0
    #loops the user until a proper player name is inputed 
    while askplayer < 1:
        playerName = input("Player:")
        if playerName in set(df_player_info['fullName']):
            #finds the row that the inputed players data is on and adds it to a list 
            skater_row = df_player_info[df_player_info['fullName'] == playerName].index.tolist()
            #locates the player_id of the inputed player from the row its on and the column its in
            new_skater_id = df_player_info.at[skater_row[0], 'player_id']
            #breaks the loop
            askplayer += 1
            #if the user input is not found in the fullName that means the player name was not spelled properly, or the player is not in the dataset
        elif playerName not in set(df_player_info['fullName']):
            print("Please enter a different NHL Player.")

    #loops the user until a proper goalie name is inputed 
    while askgoalie < 1:
        goalieName = input("Goalie:")
        if goalieName in set(df_player_info['fullName']):
            #finds the row that the inputed goalies data is on and adds it to a list 
            goalie_row = df_player_info[df_player_info['fullName'] == goalieName].index.tolist()
            #locates the player_id of the inputed goalie from the row its on and the column its in
            new_goalie_id = df_player_info.at[goalie_row[0], 'player_id']
            #breaks the loop
            askgoalie +=1
            #if the user input is not found in the fullName that means the goalie name was not inputed properly, or the goalie is not in the dataset
        elif goalieName not in set(df_player_info['fullName']):
            print("Please enter a different NHL Goalie.")

    #loops the user until a proper shot type is inputed 
    while askshot < 1:
        shot = input("Shot Type(WristShot, SnapShot, SlapShot, TipIn, WrapAround, Backhand):")
        '''
        this if statement adds 5 0s and a 1 to the empty array depending on which shot type is inputed.
        depending on the order of the 6 numbers, it tells the code what shot was used 
        '''
        if shot == "WristShot":
            #list of data that tells the code that a wrist shot was used
            wrist_data = [1, 0, 0, 0, 0, 0]
            #adds the data to the empty array
            new_data[0].extend(wrist_data)
            #breaks the loop
            askshot += 1
        elif shot == "SnapShot":
            #list of data that tells the code that a snap shot was used
            snap_data = [0, 1, 0, 0, 0, 0]
            #adds the data to the empty array
            new_data[0].extend(snap_data)
            #breaks the loop
            askshot += 1
        elif shot == "SlapShot":   
            #list of data that tells the code that a slap shot was used
            slap_data = [0, 0, 1, 0, 0, 0]
            #adds the data to the empty array
            new_data[0].extend(slap_data)
            #breaks the loop
            askshot += 1
        elif shot == "TipIn":
            #list of data that tells the code that a Tip In shot was used
            tip_data = [0, 0, 0, 1, 0, 0]
            #adds the data to the empty array
            new_data[0].extend(tip_data)
            #breaks the loop
            askshot += 1
        elif shot == "WrapAround":
            #list of data that tells the code that a Wrap Around shot was use
            wrap_data = [0, 0, 0, 0, 1, 0]
            #adds the data to the empty array
            new_data[0].extend(wrap_data)
            #breaks the loop
            askshot += 1
        elif shot == "Backhand":
            #list of data that tells the code that a backhand shot was use
            back_data = [0, 0, 0, 0, 0, 1]
            #adds the data to the empty array
            new_data[0].extend(back_data)
            #breaks the loop
            askshot += 1
        #this last elif statement is if none of the inputs match any of the expected inputs
        elif shot != "WristShot" or shot != "SnapShot" or shot != "SlapShot" or shot != "TipIn" or shot != "WrapAround" or shot != "Backhand":
            print("Please input one of the listed shot types.")

    #the middle of the nets cords
    middle_goal_x = 86 
    middle_goal_y = 0
    #finds the distance of the shots origin to the net using pythagorean theorem
    new_distance = math.sqrt((middle_goal_x - new_data[0][0])**2 + (middle_goal_y - new_data[0][1])**2)
    #adds the distance to the empty array
    new_data[0].append(new_distance)

    '''
    To calculate the angle of the shot, I created a right triangle using the shots origin,
    the net and a point on the x axis. Then using trig, I found the angle.
    '''
    #finds the distance between the y value of the input data and the x axis
    adjacent = (middle_goal_y - new_data[0][1])
    '''
    Sets angle to 0 if y = 0. 
    If y = 0 then one of the triangles side lengths would be 0. 
    Therefore the shape cant be a triangle and trig wouldn't work 
    '''
    if adjacent == 0:
        new_angle = 0
    else:
        #finds the angle of the shot using tan(angle) = opp/adj
        new_angle = math.fabs(math.atan((middle_goal_x - new_data[0][0]) / adjacent))

    #adds the angle to the empty array
    new_data[0].append(new_angle)

    #finds what row that the inputed player's stats are on and adds it to a list
    skrow = df_game_skater_stats[df_game_skater_stats['skater_id'] == new_skater_id].index.tolist()
    grow = df_game_goalie_stats[df_game_goalie_stats['goalie_id'] == new_goalie_id].index.tolist()
    new_goals = df_game_skater_stats.at[skrow[0], 'goals']
    new_shots = df_game_skater_stats.at[skrow[0], 'shots']
    new_assists = df_game_skater_stats.at[skrow[0], 'assists']
    new_timeOnIce = df_game_skater_stats.at[skrow[0], 'timeOnIce']
    
    #finds what row that the inputed goalie's savePercentage is on and adds it to a list
    new_savePercentage = df_game_goalie_stats.at[grow[0], 'savePercentage']

    #adds all of thedata that was just found to the empty array
    new_data[0].append(new_goals)
    new_data[0].append(new_shots)
    new_data[0].append(new_assists)
    new_data[0].append(new_timeOnIce)
    new_data[0].append(new_savePercentage)

    #predicts if a goal would be scored based on the data added to the empty array
    new_pred = (lr_model.predict_proba(new_data)[:,1]>=0.575).astype(int)
    new_prob = lr_model.predict_proba(new_data)
    #if statement that prints goal if the model returns 1 or no goal if the model returns 0 along with the percentage of the goal going in
    if new_pred[0] == 1:
        print("goal with a ", new_prob[0][1], "%", "chance of going in")
    elif new_pred[0] == 0:
        print("no goal with a ", new_prob[0][1], "%", "chance of going in")
