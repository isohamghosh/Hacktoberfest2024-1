import pandas as pd
import numpy as np

rating_data  = {
    'User1' : [5, 3, 0, 1, 0],
    'User2' : [4, 0, 0, 1, 1],
    'User3' : [1, 1, 0, 5, 5],
    'User4' : [1, 0, 4, 4, 4]
}

rating_df = pd.DataFrame(rating_data, index=['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5']).T

rating_matrix = rating_df.values

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.002, beta=0.02):
    Q = Q.T
    count = 0
    for step in range(steps):
        for i in range(len(R)):             # Iterate over row
           for j in range(len(R[i])):       # Iterate over column
               if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])   # Calculate error
                    for k in range(K):
                        P[i][k] += alpha*(2*eij*Q[k][j]-beta * P[i][k])
                        Q[k][j] += alpha*(2*eij*P[i][k]-beta * Q[k][j])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e+= (R[i][j] - np.dot(P[i, :], Q[:, j]))**2
                    for k in range(K):
                        e += (beta/2) * (P[i][k]**2 + Q[k][j]**2)
        count += 1
        if e < 0.001:  # Stop if the error is very small
            print("no. of steps:", count)
            break

    return P, Q.T

K = 2
num_users, num_movies = rating_matrix.shape

P = np.random.rand(num_users, K)
Q = np.random.rand(num_movies, K)

nP, nQ = matrix_factorization(rating_matrix, P, Q, K)

predicted_ratings_matrix = np.dot(nP, nQ.T)

predicted_ratings_df = pd.DataFrame(predicted_ratings_matrix, index=rating_df.index, columns=rating_df.columns)

print("Predicted ratings:")
print(predicted_ratings_df)

print("Actual ratings:")
print(rating_df)

