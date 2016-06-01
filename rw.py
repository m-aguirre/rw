# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:29:31 2016

@author: matthewaguirre
"""

import numpy as np

import pandas as pd

df = pd.read_csv('/Users/matthewaguirre/Desktop/nba_season.csv')

#drop rows we don't need
df = df.drop(['Date'], 1)
df = df.drop(['Start (ET)'],1)
df = df.drop(['Unnamed: 2'], 1)
df = df.drop(['Unnamed: 7'], 1)
df.convert_objects(convert_numeric=True)


i = 0
for i in range(0,1230): #gets the winner of each game
   if df['PTS1'][i] > df['PTS2'][i]:
        #print df['team_1'][i]
        df['Winner'][i] = df['team_1'][i]
   else:
        df['Winner'][i] = df['team_2'][i]

        
df1 = df[['team_1']] #subdataframe to build team dictionary

#builds a dictionary that maps each team name to an integer value, 0-29
teams = {team:i for i, team in enumerate(df['team_1'].unique())}

matrix = np.zeros(shape = (30,30)) #initialize our 30 x 30 matrix of NBA teams

point_diff = np.zeros(30) #point differential vector for Massey

prob_vec = np.zeros(30)

prob_vec[0] = 1
    
z = 0
i = 0
j = 0
pt = df['PTS1'] - df['PTS2']
for z in range(0, 1230):
    if df['team_1'][z] == df['Winner'][z]:

        matrix[teams[df['team_1'][z]],teams[df['team_1'][z]]] += 1.0

    elif df['team_1'][z] != df['Winner'][z]:

        matrix[teams[df['team_2'][z]],teams[df['team_1'][z]]] += 1.0
        matrix[teams[df['team_2'][z]],teams[df['team_2'][z]]] += 1.0 




def divis(C): # divides each entry in the matrix by 82, the number of
                # games played by each team
    i = 0
    j = 0        
    for i in range(0,30):
        row = C[i]
        for j in range(0,30):
            row[j] = row[j]/82.0
    print "done"
    return C
  
index = 0



#sample matrix for Markov ranker
C=np.array([[12.0/30.0, 7.0/30.0, 2.0/30.0, 3.0/30.0], 
           [3.0/30.0, 13.0/30.0, 4.0/30.0, 6.0/30.0],
           [8.0/30.0, 6.0/30.0, 19.0/30.0, 5.0/30.0],
           [7.0/30.0, 4.0/30.0, 5.0/30.0, 16.0/30.0]])


pvec = np.array([0.25, 0.25, 0.25, 0.25])

pvec2 = np.array([1,0,0,0])






def randomWalk(C, n, p):
    
    x = len(C)
    
    zero_vec = np.zeros(x)
    
    for i in range(x):
        
        zero_vec[i] = 1.0/(C[:,i].sum())
    
    diag_mx = np.diag(zero_vec)
    
    q = np.linalg.matrix_power(C.dot(diag_mx),n).dot(p)
   
    
    return   (1/q.sum())*q
    
def mark_rank(C, n, p): #performs and prints markov ranking on sample matrix
    rank_vector = randomWalk(C, n, p)
    print rank_vector    
    
def ranker(C, n, p, teams): #markov ranking on NBA data
    
    C = divis(C)
    rank_vector = randomWalk(C, n, p)
    print rank_vector
    x = len(rank_vector) 
    i = 0
    
    while i < x:
        
        y = max(rank_vector)
        for j in range(0, x):
            f = 0
            if rank_vector[j]==y:
                #for f in range (0,30):
                tm = teams.keys()[teams.values().index(j)]
                rank_vector[j] = 0
                
        print tm      
   #     print y 
        i = i + 1
        
    
def rank(C, n, P):
    
    rank_vector = randomWalk(C, n, P)
    
    arr = pArray(rank_vector)    
    
    return arr        
        

def pArray(rank):
    
    row = [0,0,0,0]
    array = np.zeros(shape = (4,4))
    num_games = 20    
    x = len(rank)
    print x
    i = 0
    j = 0
    for i in range(0, len(rank)):
        
        for j in range(0, len(rank)):
            
            A = rank[i]
        
            B = A/(A + rank[j])
            
            row[j] = B
            
        array[i] = row
        row = [0,0,0,0]
        
    return array
    

def season(prob_array):
    num_games = 20
    i =0
    j = 0
    print len(prob_array)
    for i in range(0, len(prob_array[i])):
        
        row = prob_array[i]
        
        for j in range(0, len(row)):
            
            if (j == i):
                continue
            else:          
                row[j] = game(row[j], num_games) 
    
    prob_array = wins(prob_array, num_games)
    #at this point we have a matrix with the total number of games won on the diagonal
    # and the number of games lost to each team in the columns    
    
    #now we just need to divide each entry by the number of games played:
    i=0
    j=0
    
    for i in range(0, len(prob_array)):
        row = prob_array[i]       
        
        for j in range(0, len(row)):
            
            if(j == i): #we're on the diagonal
                row[i] = row[i] / float(num_games*len(prob_array - 1))
            else:
                row[j] = row[j] / float(num_games)
            
                
        
        
    
    return prob_array
            
        
     
def wins(season_array, num_games): #gets the diagonal for our season_array
                                   #this is the number of games won/total games played 
    games_played = num_games * (len(season_array) - 1)
    
    print games_played
    
    i = 0
    j = 0
    h =0
    losses = 0
    row = season_array[i]
    
    for i in range(0, len(season_array)):
        
        current_row = season_array[i]
        
        for j in range(0, len(season_array)):
            
            if(j==i):
                continue
            else:
                row = season_array[j]
                losses = losses + row[i]
        
        
        for h in range(0, len(season_array)):
            if(h==i):
                current_row[i] = games_played - losses
            else:
                continue
        
        losses = 0
        
    return season_array
        

    
def game(B, num_games): #gets the number of losses of team A to B
        
        A_losses = 0
        
        for i in range(0, num_games +1):
            
            x = np.random.random_sample()
        
            if x < B:
                A_losses = A_losses + 1
            
            
 
        return A_losses


def colley(num_games, num_teams, games, teams, df): #performs Colley method on matrix
    
    C = np.zeros( shape = (30,30))
    b = np.zeros(30)
    
    C = mx(C, df,teams)    
    b = b_vec(b, df)
    
#now we have all the componts and can solve Cr = b
    
    r = np.linalg.solve(C,b)    
    
    i = 0
    x = len(r)
    while i < x:
        
        y = max(r)
        for j in range(0, x):
            f = 0
            if r[j]==y:
                #for f in range (0,30):
                tm = teams.keys()[teams.values().index(j)]
                r[j] = 0
                
        print tm      
        i = i + 1
    
    return 
    

def mx(C, df, teams): #gets Colley matrix for ranking
    z = 0
    i = 0
    j = 0
    wins = np.zeros(30)
    losses = np.zeros(30)

    for i in range (0,30):
        for j in range (0,30):
            if i==j:
                C[i,j] = 2 + 82 # 2 + number of games each team plays
            else:
                C[i,j] = -1.0*(df[i,j] + df[j,i]) # -number of games played between teams
    

    return C

def M_mx(C, df, teams): #gets Massey matrix for ranking
    i = 0
    j = 0

    for i in range (0,30):
        for j in range (0,30):
            if i==j:
                C[i,j] = 82 #number of games each team plays, differs from Colley matrix
            else:           # since C = M + 2I
                C[i,j] = -1.0*(df[i,j] + df[j,i]) # -number of games played between teams
    
    h = 0
    for h in range (0,30):
        C[29,h] = 1
    
    print C
    

    return C
def b_vec(b, df): # gets win - loss vector

    vec = b
    wins = np.zeros(30)
    losses = np.zeros(30)
    i = 0
    j = 0
    
    for i in range(0,30):
        for j in range(0,30):
            if i == j:
                wins[i] = df[i,j]
            else:
                losses[i] = losses[i] + df[i,j]
    
    vec = wins - losses
    

    return vec
            
def massey(matrix, point_diff, pt, df): # Massey ranking method
 
     point_diff = diff(point_diff, pt, df)
     M = np.zeros(shape = (30,30))
     M = M_mx(M, matrix,teams)
     print point_diff
     
     r = np.linalg.solve(M,point_diff)    
     i = 0
     x = len(point_diff)
     while i < x:
        
         y = max(r)
         for j in range(0, x):
             if r[j]==y:
                 tm = teams.keys()[teams.values().index(j)]
                 r[j] = -10000
                
         print tm      

         i = i + 1
    
     
def diff(point_diff, pt, df):
    z = 0
    
    for z in range (0,1230):
        if df['team_1'][z] == df['Winner'][z]:
             point_diff[teams[df['team_1'][z]]] += abs(pt[z]) 
        elif df['team_1'][z] != df['Winner'][z]:
             point_diff[teams[df['team_1'][z]]] -= abs(pt[z])
             
    point_diff[29] = 0
    
    print point_diff
    

    return point_diff
    
    