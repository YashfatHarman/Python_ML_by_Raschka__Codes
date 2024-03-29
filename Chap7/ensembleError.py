from scipy.misc import comb
import math

#function to compute the error of an ensemble method
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier/2.0)
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)
    pass

#print(ensemble_error(n_classifier = 11, error = 0.25))


#compute ensemble error rates for a range of different base errors from 0.0 to 1.0 to visualize the relationship between ensemble and base errors in a line graph
import numpy as np

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

import matplotlib.pyplot as plt
plt.plot(error_range, ens_errors, label = "Ensemble error", linewidth = 2)
plt.plot(error_range, error_range, linestyle = "--", label = "Base error", linewidth = 2)
plt.xlabel("Base error")
plt.ylabel("Base/Ensemble error")
plt.legend(loc="upper left")
plt.grid()
plt.show()
    


