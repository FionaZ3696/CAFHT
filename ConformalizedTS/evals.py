
import numpy as np
import pandas as pd

def sliding_window_average(arr, band_width):
        averages = []
        for i in range(len(arr)):
            if i - band_width < 0:
                start = 0
                end = i + band_width + 1
            elif i + band_width > len(arr) - 1:
                start = i - band_width
                end = len(arr) 
            else:
                start = i - band_width 
                end = i + band_width + 1
            window = arr[start : end]
            average = np.mean(window)
            averages.append(average)
        return averages
    

def evaluation(test_pred, test_true, PI, method, data_gen, K, band_width, hard_idx = None):

    n_test = test_pred.shape[0]
    horizon = test_pred.shape[1]

    size = np.empty((n_test, horizon))
    coverage = np.empty((n_test, horizon))

    local_coverage = np.empty((n_test, horizon))
    results = pd.DataFrame({})

    for idx in range(n_test):
        for h in range(horizon):
            test_ = test_true[idx, h]
            pred_low_ = PI[idx][h][0]
            pred_high_ = PI[idx][h][1]
            cover = int((test_ <= pred_high_) and (test_ >= pred_low_))
            size[idx,h] = pred_high_- pred_low_
            coverage[idx,h] = cover

    simutaneous_coverage = np.sum(coverage, axis = 1) == horizon
    at_most_K_coverage = np.sum(coverage, axis = 1) >= horizon - K
    average_coverage = np.mean(np.mean(coverage, axis = 1))

    def sliding_window_average(arr, band_width):
        averages = []
        for i in range(len(arr)):
            if i - band_width < 0:
                start = 0
                end = i + band_width + 1
            elif i + band_width > len(arr) - 1:
                start = i - band_width
                end = len(arr)
            else:
                start = i - band_width
                end = i + band_width + 1
            window = arr[start : end]
            average = np.mean(window)
            averages.append(average)
        return averages

    for i in range(n_test):
        cov_temp = coverage[i]
        local_coverage[i,:] = sliding_window_average(cov_temp, band_width)

    local_coverage = np.mean(local_coverage)


    results['Simutaneous coverage'] = [np.mean(simutaneous_coverage)]
    if hard_idx is not None:
      
      results['Conditional coverage (hard)'] = [np.mean(simutaneous_coverage[hard_idx == True])]
      results['Conditional coverage (easy)'] = [np.mean(simutaneous_coverage[hard_idx == False])]

    results['K-times-most coverage'] = [np.mean(at_most_K_coverage)]
    results['local (sliding window) coverage'] = [np.mean(local_coverage)]
    results['Average coverage'] = [np.mean(average_coverage)]
    results['Size'] = [np.mean(size)]
    results['Method'] = [method]
    results['data'] = [data_gen]

    return results



def evaluation_multivariate(test_pred, test_true, PI, method, data_gen, hard_idx, easy_idx):

    n_test = test_pred.shape[0]
    horizon = test_pred.shape[1]

    assert len(test_pred.shape) == 3, "wrong data dimension, expect 3D"
    ndim = test_pred.shape[2]

    size = np.empty((n_test, horizon, ndim))
    coverage = np.empty((n_test, horizon, ndim))

    local_coverage = np.empty((n_test, horizon, ndim))
    results = pd.DataFrame({})

    for idx in range(n_test):
        for h in range(horizon):
          for d in range(ndim):
            test_ = test_true[idx, h, d]
            pred_low_ = PI[idx][h][0][d]
            pred_high_ = PI[idx][h][1][d]
            cover = int((test_ <= pred_high_) and (test_ >= pred_low_))
            size[idx,h, d] = pred_high_- pred_low_
            coverage[idx,h, d] = cover
    simutaneous_coverage = np.sum(np.sum(coverage, axis = 1) == horizon, axis = 1) == ndim # np.sum(coverage, axis = 1) == horizon
    cond_coverage_hard = np.sum(np.sum(coverage[hard_idx, :], axis = 1) == horizon, axis = 1) == ndim
    cond_coverage_easy = np.sum(np.sum(coverage[easy_idx, :], axis = 1) == horizon, axis = 1) == ndim
    average_coverage = np.mean(np.mean(np.mean(coverage, axis = 1), axis = 1))

    results['Simutaneous coverage'] = [np.mean(simutaneous_coverage)]
    results['Conditional coverage-hard'] = [np.mean(cond_coverage_hard)]
    results['Conditional coverage-easy'] = [np.mean(cond_coverage_easy)]
    results['Average coverage'] = [np.mean(average_coverage)]
    results['Size'] = [np.mean(size)]
    results['Method'] = [method]
    results['data'] = [data_gen]

    return results