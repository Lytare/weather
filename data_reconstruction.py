import numpy as np
import pandas as pd
import random

if __name__ == "__main__":

    path_data2009 = r"D:\weather\data\wind_turbines\Wind Spatio-Temporal Dataset1(2009, Ave).csv"
    path_data2010 = r"D:\weather\data\wind_turbines\Wind Spatio-Temporal Dataset1(2010, Ave).csv"

    data2009 = pd.read_csv(path_data2009)
    data2010 = pd.read_csv(path_data2010)

    data = pd.concat([data2009, data2010])

    N = data.shape[1]  # number of measurement locations

    # knock out entries (simulating missing buoy data)
    random.seed(42)

    # leave untouched for forecasting purposes: 24 (hourly) * 5 (5 days) of data
    N_safe = 24 * 3
    data_faulty = data[:-N_safe].copy()
    data_safe = data[-N_safe:].copy()

    # e.g. 10 have missing data
    for b in random.sample(range(1, N), 10):
        samples = random.randint(5, 50) * 24  # missing between 5 and 50 days worth of data
        st = random.randint(1, data_faulty.shape[0])  # starting at random sample
        data_faulty.iloc[st:min([data_faulty.shape[0] - 1, st + samples]), b] = np.nan


    # low-rank matrix completion, based on algorithms described by Rik Voorhaar
    X = data[:-N_safe].to_numpy(copy=True)
    y = data_faulty.to_numpy(copy=True)
    rank = 10 
    epochs = 40 # rank and epoch settings seem fine and give good results; may require tweaking for proprietary data

    # decomposition
    A = np.random.normal(size=(X.shape[0], rank))
    B = np.random.normal(size=(rank, X.shape[1]))
    Omega = np.argwhere(np.isfinite(y)).T
    Omega = (Omega[0], Omega[1])

    y = y[Omega]

    def linsolve_regular(A, b, lam=1e-4):
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    
    losses = []
    for epoch in range(epochs):
        print(epoch)
        loss = np.mean(((A @ B)[Omega] - y) ** 2)
        losses.append(loss)

        # Update B
        for j in range(B.shape[1]):
            B[:, j] = linsolve_regular(A[Omega[0][Omega[1] == j]], y[Omega[1] == j])

        # Update A
        for i in range(A.shape[0]):
            A[i, :] = linsolve_regular(B[:, Omega[1][Omega[0] == i]].T, y[Omega[0] == i])


    # substitute only into missing portions of data
    y_recon = A @ B
    X_recon = data_faulty.to_numpy(copy=True)
    X_recon[np.isnan(data_faulty).to_numpy()] = y_recon[np.isnan(data_faulty).to_numpy()]

    data_recon = pd.DataFrame(data=X_recon, columns=data.columns)
    

    # merge data and push to dataset location; including untouched dataset for benchmark testing
    data_complete = pd.concat([data_recon, data_safe])
    data_complete.to_csv(os.path.join('TimeGNN', 'datasets', 'WindSTData_recon.csv'))
    data.to_csv(os.path.join('TimeGNN', 'datasets', 'WindSTData.csv'))

    # for the GNN, please call TimeGNN_train.py, which is a reconfigured branch of the third-party open-source TimeGNN
    # Xu et al., 2023 (https://doi.org/10.48550/arXiv.2307.14680)
    # Xu et al. tested their GNN on multimodal meteorology data and reported high predictive power




