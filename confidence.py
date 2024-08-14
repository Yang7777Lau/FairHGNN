import pandas as pd
import numpy as np
import seaborn as sns


data = pd.DataFrame({
    'Column1': [89.44,90.68,88.94,89.57,91.68],
    'Column2': [94.47,95.08,94.58,95.04,95.68],
    'Column3': [2.04,3.53,2.99,4.98,2.18],

})

# Define the function to calculate the confidence interval half-width
def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = sns.algorithms.bootstrap(data, func=func, n_boot=size, seed=seed)
    bounds = sns.utils.ci(bs_replicates, ci)
    return (bounds[1] - bounds[0]) / 2

# Calculate the confidence interval half-width for Column1
ci_half_width = confidence_interval(data['Column1'])
print(ci_half_width)
ci_half_width = confidence_interval(data['Column2'])
print(ci_half_width)
ci_half_width = confidence_interval(data['Column3'])
print(ci_half_width)

