import numpy as np
import random

###################################################################################################################
#                                             Data model                                                          #
###################################################################################################################

class data_gen:
  def __init__(self, N, length, hetero = True, seed = 2023, delta = 0.5, ndim = 3):
    self.N = N
    self.length = length
    self.seed = seed
    self.hetero = hetero
    self.delta = delta
    u = np.random.uniform(0,1, self.N)
    self.u = u
    self.ndim = ndim

  def generate_AR(self, noise_profile = 'static', amplitude = 1, phi = [0.9, 0.1, -0.2], noise_level = 10, return_index = False):
    data = np.zeros((self.N, self.length, self.ndim))
    order = len(phi)

    for i in range(order, self.length): # starting from the p+1th element
        for j in range(order):
            data[:,i, :] += phi[j] * data[:,i-j-1, :]
        if self.hetero:
          if noise_profile == 'dynamic':
            data[self.u <= self.delta, i, :] += np.random.normal(0, i*noise_level, (sum(self.u <= self.delta), self.ndim))
            data[self.u > self.delta, i, :] += np.random.normal(0, i, (sum(self.u > self.delta), self.ndim))
          elif noise_profile == 'static':
            data[self.u <= self.delta, i, :] += np.random.normal(0, amplitude*noise_level, (sum(self.u <= self.delta), self.ndim))
            data[self.u > self.delta, i, :] += np.random.normal(0, amplitude, (sum(self.u > self.delta), self.ndim))
        else:
          if noise_profile == 'dynamic':
            data[:,i, :] += np.random.normal(0, i , (self.N, self.ndim))
          elif noise_profile == 'static':
            data[:,i,:] += np.random.normal(0, amplitude,(self.N, self.ndim))  # Add some random noise
    if return_index:
      u_ = np.array(self.u <= self.delta)
      return data, u_

    return data

  def generate_random_noise(self, data = None, amplitude = 1):
    return data + np.random.normal(0, amplitude, (self.N, self.length, self.ndim)) if data else np.random.normal(0, amplitude, (self.N, self.length, self.ndim))

  def generate_AR_seasonality(self, amplitude, period, random_start = True, data = None, season_hetero = True):
    ###### TO DO : extend to multidimensional ########
    if data is None:
      data = self.generate_AR()
    starting_point = np.random.randint(0, self.length, size = (self.N, self.ndim)) if random_start else np.zeros((self.N, self.ndim))
    for i in range(self.N):
      if season_hetero and self.u[i] <= self.delta:
        seasonality_data = amplitude*10 * np.sin(2 * np.pi * (np.arange(self.length)+ starting_point[i]) / period)
      else:
        seasonality_data = amplitude * np.sin(2 * np.pi * (np.arange(self.length)+ starting_point[i]) / period)
      data[i] += seasonality_data
    return data

  def generate_AR_random_peaks(self, num_peaks, max_amplitude = 10, data = None, peak_hetero = True):
    ###### TO DO : extend to multidimensional ########
    if data is None:
      data = self.generate_AR()
    peak_positions = np.array([np.random.choice(range(self.length), num_peaks, replace=False) for _ in range(self.N)])
    peak_amplitudes = np.empty((self.N, num_peaks))
    for i in range(self.N):
      if peak_hetero and self.u[i] <= self.delta:
        for j in range(num_peaks):
          peak_amplitudes[i, j] = max_amplitude*10 if random.randint(0, 1) == 0 else -max_amplitude*10
      else:
        for j in range(num_peaks):
          peak_amplitudes[i, j] = max_amplitude if random.randint(0, 1) == 0 else -max_amplitude
    for i in range(self.N):
        for j in range(num_peaks):
            data[i, peak_positions[i, j]] += peak_amplitudes[i, j]
    return data

  def generate_AR_volClus(self, max_amplitude = 10, data = None, vol_hetero = True):
    ###### TO DO : extend to multidimensional ########
    if data is None:
      data = self.generate_AR()

    start_positions = np.random.randint(0, self.length, size=self.N)
    span = np.random.randint(1, self.length/3, size=self.N)
    end_positions = np.minimum(start_positions+span, self.length -1)

    for i in range(self.N):
      if vol_hetero and self.u[i] <= self.delta:
        data[i, start_positions[i]:end_positions[i]] += np.random.normal(0, max_amplitude*10, end_positions[i] - start_positions[i])
      else:
        data[i, start_positions[i]:end_positions[i]] += np.random.normal(0, max_amplitude, end_positions[i] - start_positions[i])
    return data
