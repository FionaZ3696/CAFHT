import numpy as np
from scipy.stats.mstats import mquantiles
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
from ConformalizedTS.utils import trimming, saturation_fn_log
from third_party.theory import inv_hybrid
import numpy.linalg as la

###################################################################################################################
#                                                    ACI                                                          #
###################################################################################################################
class Adaptive_Conformal_Inference:
    """
    ACI method for ***single*** time series sequence
    """
    def __init__(self, alpha, verbose = True):
        self.alpha = alpha
        self.verbose = verbose


    def add_warm_start(self, pred, true, gamma):

        horizon = pred.shape[0]
        scores = la.norm(pred - true, np.inf, axis = 1)# np.abs(pred - true)

        alphat = self.alpha
        adaptErrSeq = np.repeat(-np.inf, horizon)

        for t in range(horizon):

            recentscores = scores[0:t] if t != 0 else scores[0]
            testscore = scores[t]

            if alphat >= 1:
                adaptErrSeq[t] = 1
            elif alphat <= 0:
                adaptErrSeq[t] = 0
            else:
                adaptErrSeq[t] = sum(testscore > mquantiles(recentscores, prob = 1-alphat))
            alphat += gamma*(self.alpha - adaptErrSeq[t])

        # print('Finish warmstart.. alphat {}'.format(alphat))
        return alphat, recentscores

    def generate_warmstart_data(self, n, first_n = 5, seed = 123, ndim = 1):
        warmstart_data = np.random.uniform(-0.05,0.05, (n, first_n*2, ndim))
        warmstart_data = [warmstart_data[:, :first_n, :], warmstart_data[:, first_n:, :]]
        return warmstart_data

    def predict_intervals(self, test_pred, test_true, gamma = 0.1, warm_start = True, first_n = 5, q0 = None, y_trim = None, seed = 123): #
        """
        predict intervals for multiple test sequences, one at a time
        Note: we don't need to refit the model here as we are working with multiple time sequences
        warmstart_data: [pred, true] of n dimension
        """
        horizon = test_pred.shape[1]
        ndim = test_pred.shape[2]
        n_test = test_pred.shape[0] # total number of sequences

        pred_intervals = np.empty((n_test, horizon, 2, ndim))

        if y_trim:
          assert isinstance(y_trim, list) and len(y_trim) == 2, "Need to input y_trim as a list with two elements"

        if warm_start:
            warmstart_data = self.generate_warmstart_data(n_test, first_n = first_n, seed = seed, ndim = ndim)

        for k in tqdm(range(n_test), disable = self.verbose == False):

            alphat = self.alpha

            if warm_start:
                alphat, recentscores = self.add_warm_start(warmstart_data[0][k], warmstart_data[1][k], gamma = gamma)

            y_pred = test_pred[k]
            y_true = test_true[k]
            scores = la.norm(y_pred - y_true, np.inf, axis = 1) # np.abs(y_pred - y_true)

            alphaSequence = np.repeat(-np.inf, horizon)
            adaptErrSeq = np.repeat(-np.inf, horizon)

            for t in range(horizon):

                calibscores = np.concatenate((recentscores, scores[0:t])) if warm_start else scores[0:t] if t != 0 else scores[0]
                testscore = scores[t]
                
                

                if alphat >= 1:
                    adaptErrSeq[t] = 1
                    pred_low, pred_high = 0, 0

                elif alphat <= 0:
                    adaptErrSeq[t] = 0
                    pred_low, pred_high = -np.inf, np.inf

                else:
                    adaptErrSeq[t] = sum(testscore > mquantiles(calibscores, prob = 1-alphat))
                    scores_calibrated = mquantiles(calibscores, prob=1-alphat)
                    pred_low = y_pred[t] - scores_calibrated[0]
                    pred_high = y_pred[t] + scores_calibrated[0]

                if y_trim and alphat < 1:
                    pred_low, pred_high = trimming(y_trim, pred_low, pred_high)

                pred_intervals[k, t, 0, :] = pred_low
                pred_intervals[k, t, 1, :] = pred_high

                alphaSequence[t] = alphat
                alphat += gamma*(self.alpha - adaptErrSeq[t])

        return pred_intervals
      
      
###################################################################################################################
#                                            PID Control                                                          #
###################################################################################################################
class Conformal_PID_control:
    """
    ACI method for ***single*** time series sequence
    """
    def __init__(self, alpha, verbose = True):
        self.alpha = alpha
        self.verbose = verbose

    def predict_intervals(self, test_pred, test_true, gamma = 0.1, q0 = 0.01, y_trim = None, seed = 123, ahead = 1, proportional_lr = True,
                          Csat = 0.5, KI = 1, integrate = False): #
        horizon = test_pred.shape[1]
        ndim = test_pred.shape[2]
        n_test = test_pred.shape[0] # total number of sequences

        pred_intervals = np.empty((n_test, horizon, 2, ndim))
        if y_trim:
          assert isinstance(y_trim, list) and len(y_trim) == 2, "Need to input y_trim as a list with two elements"

        for k in tqdm(range(n_test), disable = self.verbose == False):
            y_pred = test_pred[k]
            y_true = test_true[k]
            scores = la.norm(y_pred - y_true, np.inf, axis = 1) # np.abs(y_pred - y_true)
            qs = np.zeros((horizon,))
            qs[0] = q0
            qts = np.zeros((horizon,))
            qts[0] = q0
            integrators = np.zeros((horizon,))
            scorecasts = np.zeros((horizon,))
            covereds = np.zeros((horizon,))

            for t in range(horizon):
              lr_t = gamma * (scores[0:t].max() - scores[0:t].min()) if proportional_lr and t > 0 else gamma
              covereds[t] = qs[t] >= scores[t]
              grad = self.alpha if covereds[t] else -(1-self.alpha)
              integrator_arg = (1-covereds)[:t].sum() - (t)*self.alpha
              integrator = saturation_fn_log(integrator_arg, t, Csat, KI)
              if t < horizon - 1:
                qts[t+1] = qts[t] - lr_t * grad
                integrators[t+1] = integrator if integrate else 0
                qs[t+1] = np.maximum(qts[t+1] + integrators[t+1], 0)

              pred_low = y_pred[t] - qs[t]
              pred_high = y_pred[t] + qs[t]

              if y_trim:
                pred_low = np.maximum(pred_low, y_trim[0])
                pred_high = np.minimum(pred_high, y_trim[1])

              pred_intervals[k, t, 0, :] = pred_low
              pred_intervals[k, t, 1, :] = pred_high
            # pdb.set_trace()
        return pred_intervals
      
###################################################################################################################
#                                          Standard conformal inference                                           #
###################################################################################################################

class Split_Conformal:
    def __init__(self, alpha, horizon, ndim = 1, bonf_correction = True):
        self.alpha = alpha
        self.horizon = horizon
        self.bonf_correction = bonf_correction
        self.ndim = ndim

    def calibrate(self, calib_pred, calib_true):
        cal_scores = np.abs(calib_pred - calib_true)
        cal_scores = la.norm(cal_scores, np.inf, axis=2)

        n_cal = cal_scores.shape[0]
        if self.bonf_correction:
            level_adjusted = (1.0 - self.alpha/self.horizon)*(1.0 + 1.0/float(n_cal))
            # level_adjusted = np.clip(level_adjusted, 1.0/n_cal, 1)
        else:
            level_adjusted = (1.0 - self.alpha)*(1.0 + 1.0/float(n_cal))
        print('adjusted alpha {}'.format(level_adjusted))
        inf_nums = np.full((1, cal_scores.shape[1]), np.inf)
        cal_scores_aug = np.append(cal_scores, inf_nums, axis=0)
        self.scores_calibrated = mquantiles(cal_scores_aug, prob=level_adjusted, axis = 0).reshape(-1,1)

    def predict(self, test_pred, y_trim = None):

        n_test = test_pred.shape[0]
        pred_intervals = np.empty((n_test, self.horizon, 2, self.ndim))


        if y_trim:
          assert isinstance(y_trim, list) and len(y_trim) == 2, "Need to input y_trim as a list with two elements"


        for h in range(self.horizon):
            pred_low = test_pred[:, h, :] - self.scores_calibrated[h]
            pred_high = test_pred[:, h, :] + self.scores_calibrated[h]

            if y_trim:
                pred_low = np.maximum(pred_low, y_trim[0])
                pred_high = np.minimum(pred_high,y_trim[1])

            pred_intervals[:, h, 0, :] = pred_low
            pred_intervals[:, h, 1, :] = pred_high


        return pred_intervals


###################################################################################################################
#                                                Max Calibration                                                  #
###################################################################################################################

class Max_calibrate:
    def __init__(self, alpha, horizon, normalize, ndim = 1):
        self.alpha = alpha
        self.horizon = horizon

        assert len(normalize) == horizon, "Normalization length not equal to the prediction horizon"
        self.normalize = normalize

        self.ndim = ndim

    def calibrate(self, calib_pred, calib_true):

        cal_scores = la.norm(np.abs(calib_pred - calib_true)/self.normalize, np.inf, axis = 2)
        cal_scores = np.max(cal_scores, axis = 1)
        n_cal = len(cal_scores)

        level_adjusted = (1.0 - self.alpha)*(1.0 + 1.0/float(n_cal))
        print('adjusted alpha {}'.format(level_adjusted))
        cal_scores_aug = np.append(cal_scores, np.inf)

        self.scores_calibrated = np.quantile(cal_scores_aug, level_adjusted, interpolation='higher') 
        
        # mquantiles(cal_scores_aug, prob=level_adjusted)


    def predict(self, test_pred, y_trim = None):

        n_test = test_pred.shape[0]
        pred_intervals = np.empty((n_test, self.horizon, 2, self.ndim))

        if y_trim:
          assert isinstance(y_trim, list) and len(y_trim) == 2, "Need to input y_trim as a list with two elements"

        for h in range(self.horizon):
          for d in range(self.ndim):
            pred_low = test_pred[:, h, d] - self.scores_calibrated*self.normalize[h, d]
            pred_high = test_pred[:, h, d] + self.scores_calibrated*self.normalize[h, d]

            if y_trim:
                pred_low = np.maximum(pred_low, y_trim[0])
                pred_high = np.minimum(pred_high,y_trim[1])

            pred_intervals[:, h, 0, d] = pred_low
            pred_intervals[:, h, 1, d] = pred_high


        return pred_intervals



###################################################################################################################
#                     Conformal Adaptive Forecastor for Heterogeneous Trajectories (CAFHT)                        #
###################################################################################################################

class CAFHT:

  def __init__(self, alpha, gamma_grid, base_model, randomize = False, verbose = True, adaptive = True):
    self.alpha = alpha
    self.gamma_grid = gamma_grid
    self.n_gamma = len(gamma_grid)
    self.randomize = randomize
    self.verbose = verbose

    assert base_model in ['ACI', 'PID'], "the baseline method should be either ACI or PID"
    if base_model == 'ACI':
      self.base_method = Adaptive_Conformal_Inference(alpha = self.alpha, verbose = False)
    elif base_model == 'PID':
      self.base_method = Conformal_PID_control(alpha = self.alpha, verbose = False)

    self.adaptive = adaptive

  def nonconf_scores(self, bands, y_true):

    scores = []

    for idx, y in enumerate(y_true):

      pred_low = bands[idx, :, 0, :]
      pred_high = bands[idx, :, 1, :]
      size_ = pred_high - pred_low
      eps_ = np.maximum( np.maximum(0, y - pred_high), np.maximum(0, pred_low - y))
      scaled_eps = eps_/size_
      scaled_eps = la.norm(scaled_eps, np.inf, axis=1)
      if self.adaptive:
        score_ = np.max(scaled_eps)
      else:
        score_ = np.maximum(np.max(np.maximum(0, y - pred_high)), np.max(np.maximum(0, pred_low - y)))
      scores.append(score_)
    return scores

  def predict_bands_subroutine(self, y_pred, y_true, gamma, scores_calibrated, q0, seed = 123):

    PI_base = self.base_method.predict_intervals(y_pred,
                                                y_true,
                                                gamma = gamma,
                                                q0 = q0,
                                                y_trim = self.y_trim,
                                                seed = seed)
    
    size_ = PI_base[:, :, 1, :] - PI_base[:, :, 0, :] # shape n*h
    if scores_calibrated == np.inf:
      pred_low = -np.inf
      pred_high = np.inf

    else:
      if self.adaptive:
        PI_base[:, :, 1, :] += scores_calibrated * size_
        PI_base[:, :, 0, :] -= scores_calibrated * size_
        # with warnings.catch_warnings(record=True) as w:
        #   new_up = PI_ACI[:, :, 1, :] + scores_calibrated * size_
        #   new_low = PI_ACI[:, :, 0, :] + scores_calibrated * size_
        #   # PI_ACI[:, :, 1, :] += scores_calibrated * size_
        #   # PI_ACI[:, :, 0, :] -= scores_calibrated * size_
        #   if len(w) > 0:
        #     pdb.set_trace()
      else:
        PI_base[:, :, 1, :] += scores_calibrated
        PI_base[:, :, 0, :] -= scores_calibrated

      pred_low = PI_base[:, :, 0, :]
      pred_high = PI_base[:, :, 1, :]

    if self.y_trim:
      pred_low = np.maximum(pred_low, self.y_trim[0])
      pred_high = np.minimum(pred_high, self.y_trim[1])

    return pred_low, pred_high


  def calibrate(self, y_pred, y_true, gamma, q0, seed):

    PI_base = self.base_method.predict_intervals(y_pred,
                                                y_true,
                                                gamma = gamma,
                                                q0 = q0,
                                                y_trim = self.y_trim,
                                                seed = seed)

    scores = self.nonconf_scores(PI_base, y_true)
    n_cal = len(scores)

    if self.randomize:
        noise = np.random.uniform(low=0.0, high=0.001, size=n_cal)
        scores += np.multiply(scores, noise)

    level_adjusted = (1.0 - self.alpha)*(1.0 + 1.0/float(n_cal))
    scores_calibrated = np.quantile(scores, level_adjusted, interpolation='higher') 

    return scores_calibrated

  def select_gamma(self, y_pred, y_true, q0, seed = 123, return_eps = False):

    band_width = []
    scores_calibrated = []

    for idx in tqdm(range(self.n_gamma), disable = self.verbose == False):
      gamma_ = self.gamma_grid[idx]
      scores_calibrated_ = self.calibrate(y_pred, y_true, gamma_, q0, seed)
      scores_calibrated.append(scores_calibrated_)

      pred_low, pred_high = self.predict_bands_subroutine(y_pred, y_true, gamma_, scores_calibrated_, q0)

      # Can substitute to other loss functions
      width_ = np.mean(pred_high - pred_low)
      band_width.append(width_)

    l_hat = np.argmin(band_width)
    gamma_hat = self.gamma_grid[l_hat]
    eps_hat = scores_calibrated[l_hat]

    if self.verbose:
      print("Selected gamma {}".format(gamma_hat))
      print("Calibrated absolute residual {}".format(eps_hat))

    return gamma_hat, eps_hat if return_eps else gamma_hat

  def predict_bands(self, method, calib_data, test_data, q0, y_trim = None, seed = 123, fixed_gamma = None):

    assert method in ['data_splitting', 'naive', 'theoretical_correction'], 'method invalid, pick one from [data_splitting, naive, theoretical_correction]'

    if y_trim:
      assert isinstance(y_trim, list) and len(y_trim) == 2, "Need to input y_trim as a list with two elements"
    self.y_trim = y_trim if y_trim else None

    calib_pred, calib_true = calib_data[0], calib_data[1]
    test_pred, test_true = test_data[0], test_data[1]

    n_test = test_pred.shape[0]
    horizon = test_pred.shape[1]
    ndim = test_pred.shape[2]

    pred_intervals = np.empty((n_test, horizon, 2, ndim))

    if method == 'data_splitting':

      calib_pred_1, calib_pred_2, calib_true_1, calib_true_2 = train_test_split(calib_pred, calib_true, test_size=0.5, random_state = seed)
      if fixed_gamma:
        gamma_hat = fixed_gamma
      else:
        gamma_hat, eps_hat = self.select_gamma(calib_pred_1, calib_true_1, q0, seed)

      eps_hat_hat = self.calibrate(calib_pred_2, calib_true_2, gamma_hat, q0, seed)
      if self.verbose:
          print('double-calibrated eps: {}'.format(eps_hat_hat))

      pred_low, pred_high = self.predict_bands_subroutine(test_pred, test_true, gamma_hat, eps_hat_hat, q0, seed = seed)

    else:
      if method == 'theoretical_correction':
        n_cal = len(calib_true)
        alpha2 = inv_hybrid(self.n_gamma, n_cal, self.alpha)
        alpha2 = np.clip(alpha2, 1.0/n_cal, 1)
        print('Correct miscoverage rate from {} to {}'.format(self.alpha, alpha2))
        self.alpha = alpha2

      elif method == 'naive':
        pass

      if fixed_gamma:
        gamma_hat = fixed_gamma
      else:
        gamma_hat, eps_hat = self.select_gamma(calib_pred, calib_true, q0,seed, return_eps = True)
        print('single-calibrated eps: {}'.format(eps_hat))
      pred_low, pred_high = self.predict_bands_subroutine(test_pred, test_true, gamma_hat, eps_hat, q0, seed)

    pred_intervals[:, :, 0, :] = pred_low
    pred_intervals[:, :, 1, :] = pred_high

    print('selected gamma: {}'.format(gamma_hat))
    return pred_intervals


