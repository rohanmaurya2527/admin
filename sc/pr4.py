#Code 1: Hardcoded Example
def fuzzy_logic(temperature, humidity):
    #Inference
    def temp_low(x):
        if x <= 25: return 1
        elif x > 25 and x < 50: return (50 - x)/25
        else : return 0    
    def temp_med(x):
        if x <= 25 or x >= 75: return 0
        elif x > 25 and x <= 50: return (x - 25)/25
        elif x > 50 and x < 75: return (75 - x)/25    
    def temp_high(x):
        if x <= 50: return 0
        elif x > 50 and x < 75: return (x - 50)/25
        else : return 1
    def humid_low(x):
        if x <= 25: return 1
        elif x > 25 and x < 50: return (50 - x)/25
        else : return 0    
    def humid_med(x):
        if x <= 25 or x >= 75: return 0
        elif x > 25 and x <= 50: return (50 - x)/25
        elif x > 50 and x < 75: return (75 - x)/25    
    def humid_high(x):
        if x <= 50: return 0
        elif x > 50 and x < 75: return (x - 50)/25
        else : return 1    
    #DeFuzzyfication
    def fan_speed_low():
        return (temp_low(temperature) + humid_low(humidity))/2            
    def fan_speed_med():
        return (temp_med(temperature) + humid_med(humidity))/2    
    def fan_speed_high():
        return (temp_high(temperature) + humid_high(humidity))/2    
    return [fan_speed_low(), fan_speed_med(), fan_speed_high()]
if __name__ == '__main__':
    temps = [23, 45, 56, 78]
    humids = [56, 45, 78, 78]
    for t, h in zip(temps, humids):
        result = fuzzy_logic(t, h)
        print(f'For Temperature : {t} and Humidity : {h}')
        print(f'Fan Speed --> Low : {result[0]}, Medium : {result[1]}, High : {result[-1]}')

#Code 2: On Movie Dataset
import numpy as np
import pandas as pd
data = pd.read_csv("imdb_movies.csv")
df = pd.DataFrame(data)
ratings = np.array(df['score'].values)
# Ratings for clustering
X = ratings.reshape(1, -1) # (features, samples)
import skfuzzy as fuzz
# Number of Clusters: Flop, Average, Hit
n_clusters = 3
# Apply FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X, c = n_clusters, m = 2, error = 0.005, maxiter = 1000, init = None
)
# Cluster membership for each movie
cluster_membership = np.argmax(u, axis = 0)
# Map cluster centers to labels
cluster_centers = cntr.flatten()
sorted_indices = np.argsort(cluster_centers)
labels_map = {sorted_indices[0]: 'Flop',
              sorted_indices[1]: 'Average',
              sorted_indices[2]: 'Hit'}
# Assign labels
df['Cluster'] = cluster_membership
df['Category'] = df['Cluster'].map(labels_map)
print(df[['names', 'score', 'Category']])

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
colors = ['red', 'orange', 'green']
plt.plot(ratings, u[1], label=f'{labels_map[1]}Membership', color=colors[1])
plt.title('Fuzzy C-Means Membership for IMDb Ratings')
plt.xlabel('Movie Index')
plt.ylabel('Membership Grade')
plt.legend()
plt.grid(True)
plt.show()
