import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#use pandas to load real_estate_dataset.csv
df = pd.read_csv("real_estate_dataset.csv")

n_samples, n_features = df.shape

columns = df.columns

np.savetxt("column_names.txt", columns, fmt="%s")

X = df[["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center"]]

y = df["Price"].values

print(f"Shape of X: {X.shape}\n")
print(f"data type of X: {X.dtypes}\n")


n_samples, n_features = X.shape

coefs = np.ones(n_features+1)

predictions_bydefn = X @ coefs[1:] + coefs[0]

X = np.hstack((np.ones((n_samples,1)), X))

predictions = X @ coefs

is_same = np.allclose(predictions_bydefn, predictions)

print(f"Are the predictions the same with X*coef[1:] + coefs[0] and X*coefs", is_same)

errors = y - predictions

loss_loop = 0
for i in range(n_samples):
    loss_loop += errors[i]**2
loss_loop = loss_loop/n_samples

loss_matrix = np.transpose(errors) @ errors / n_samples

is_diff = np.allclose(loss_loop, loss_matrix)
print(f"Are the loss by direct and matrix smae? {is_diff}\n")
rel_errors = errors/y
print(f"Size of errors: {errors.shape}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}")
print(f"L2 norm of relative errors: {np.linalg.norm(rel_errors)}")


# the optimization problem here is to find the coefficients that minimise the mean squared error
# this is called least sqaure problem

# the solution to find the coefficients at which the gradient of the objective function is zero
# write the loss_matrix in terms of the data and coefs

loss_matrix = (y - X @ coefs).T @ (y - X @ coefs) / n_samples

# calculate the gradient of the loss with respect to the coefficients
grad_matrix = -2 / n_samples * X.T @ (y - X @ coefs)

# Solve grad_matrix=0 for coefs
# X.T @ y = X.T @ X @ coefs
# X.T @ X @ coefs = X.T @ y. This equation is called as the Normal equation
# coefs = (X.T@X)^-1 @ X.T @ y

coefs = np.linalg.inv(X.T @ X) @  X.T @ y
np.savetxt("coefs.csv", coefs, delimiter=",")

prediction_model = X @ coefs

errors_model = y - prediction_model

rel_errors_model = errors_model/y
print(f"L2 norm of error_model: {(np.linalg.norm(rel_errors_model))}")


X = df.drop("Price", axis =1).values
y = df["Price"].values

n_samples, n_features = X.shape
print(f"Number of samples, features: {n_samples, n_features}")

X = np.hstack((np.ones((n_samples, 1)), X))

############################### inverse method ################################

coefs = np.linalg.inv(X.T @ X) @  X.T @ y
np.savetxt("coefs_all.csv", coefs, delimiter=",")

############################### QR decomposition ################################

rank_XTX = np.linalg.matrix_rank(X.T @ X)
print(f"Rank of X.T @ X: {rank_XTX}")
Q,R = np.linalg.qr(X)

np.savetxt("R.csv", R, delimiter=",")

# R*coeffs = b

sol = Q.T @ Q
np.savetxt("sol.csv", sol,delimiter=",")

# X = QR
# R*coeffs = Q.T @ y

b = Q.T @ y

coeffs_qr_loop = np.zeros(n_features+1)

for i in range(n_features, -1, -1):
    coeffs_qr_loop[i] = b[i]
    for j in range(i+1, n_features+1):
        coeffs_qr_loop[i] = coeffs_qr_loop[i] - R[i,j]*coeffs_qr_loop[j]

    coeffs_qr_loop[i] = coeffs_qr_loop[i]/R[i,i]

np.savetxt("coeffs_qr_loop.csv", coeffs_qr_loop, delimiter=",")

############################### SVD decomposition ################################

# solve normal eq using svd X = U S V^T

U, S, Vt = np.linalg.svd(X, full_matrices=False)

coeffs_svd = Vt.T @ np.diag(1/S) @ U.T @ y

coeffs_svd_pinv = np.linalg.pinv(X) @ y

np.savetxt("coeffs_svd.csv", coeffs_svd, delimiter=",")

np.savetxt("coeffs_svd_pinv.csv", coeffs_svd_pinv, delimiter=",")


# plot the data on X[:,1] vs y axis

# X_1 = X[:, 0:1]
# coeffs_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y

# X_feature = np.arange(np.min(X_1[:,1]), np.max(X_1[:,1]), 0.01)
# plt.scatter(X[:,1],y) 
# plt.plot(X_feature, X_feature * coeffs_svd[1], color="red")
# plt.xlabel('Square Feet')
# plt.ylabel("Price")
# plt.title("Price vs Square Feet")
# plt.show()


X = df["Square_Feet"].values
y = df["Price"].values

X = np.hstack((np.ones((n_samples,1)), X.reshape(-1,1)))

coeffs_1 = np.linalg.inv(X.T @ X) @ X.T @ y

predictions_1 = X @ coeffs_1

X_feature = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.01)



plt.scatter(X[:,1],y) 
plt.plot(X_feature, X_feature * coeffs_1[1] + coeffs_1[0], color="red")
plt.xlabel('Square Feet')
plt.ylabel("Price")
plt.title("Price vs Square Feet")
plt.show()

