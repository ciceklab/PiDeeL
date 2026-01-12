from sksurv.ensemble import RandomSurvivalForest
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import KFold

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(SCRIPT_DIR.parent.parent))  # scripts/

# Import repo-level config for paths
import config as repo_config

from load_targeted_data import df

task = f"rf"
c_indices = []

df3 = df.drop(columns=["duration","event"],axis=1)
X_rf = df3
y_rf = []
for i in range(len(df["event"])):
    a = (bool(df["event"][i]), float(df["duration"][i]))
    y_rf.append(a)

y_rf = np.array(y_rf, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

SEEDS = [6, 35, 81]
for SEED in SEEDS:
    i=0
    kf = KFold(n_splits=5, shuffle= True,random_state=SEED)
    for train_index, test_index in kf.split(X_rf):
        print(SEED, i)
        X_train, X_test = X_rf.iloc[train_index], X_rf.iloc[test_index]
        y_train, y_test = y_rf[train_index], y_rf[test_index]

        rsf = RandomSurvivalForest(n_estimators=500,
                                   min_samples_split=7,
                                   min_samples_leaf=10,
                                   max_features="sqrt",
                                   n_jobs=-1,
                                   random_state=SEED,
                                   verbose=0)
        rsf.fit(X_train, y_train)
        c_indices.append(rsf.score(X_test, y_test))
        i+=1
with open(repo_config.get_log_path(f"baseline/{task}", "c_indices.txt"), "w") as f:
    for c_index in c_indices:
        f.write("%f\n" % c_index)