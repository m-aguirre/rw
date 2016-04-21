# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:29:31 2016

@author: matthewaguirre
"""

import numpy as np

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
    
    
def ranker(C, n, p):
    
    rank_vector = randomWalk(C, n, p)
    
    x = len(rank_vector) 
    i = 0
    
    while i < x:
        
        y = max(rank_vector)
        for j in range(0, x):
            if rank_vector[j]==y:
                rank_vector[j] = 0
                
                
        print y 
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
                row[j] = game(row[j], num_games) #/ float(num_games) 
    
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


    