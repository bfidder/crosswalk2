import collections

class Welford:
	"""A simple Welford one-pass statistics tracker.

	There are several properties you can access after adding a set of data
	points:
	  * `xbar` - the sample mean.
	  * `var` - the sample variance (already divided by i).
	  * `i` - the last i (i.e., the number of data points).
	  * `min` - the minimum x_i that has been seen.
	  * `max` - the maximum x_i that has been seen.

	Also, to get the sample autocorrelation for a given lag, you can call
	`r(lag)`.
	"""

	def __init__(self, maxlag = 1):
		"""`maxlag` shows the maximum possible lag that this `Welford` object will track."""
		self.maxlag = max(maxlag, 1)
		self._X = collections.deque([], self.maxlag)
		self.xbar = 0
		self._v = 0
		self.i = 0
		self._W = [0] * self.maxlag
		self.min = None
		self.max = None

	@property
	def var(self):
		return self._v / self.i

	def add(self, x_i):
		"""Add a data point."""
		self.i += 1

		d = x_i - self.xbar

		if self.i == 1:
			self.min = x_i
			self.max = x_i
		else:
			if self.maxlag > 1:
				for j in range(0, min(self.i - 1, self.maxlag)):
					self._W[j] += (self.i - 1) / self.i * d * (self._X[-j - 1] - self.xbar)

			self.min = min(self.min, x_i)
			self.max = max(self.max, x_i)

		self._X.append(x_i)

		self.xbar += d / self.i
		self._v += d * d * (self.i - 1) / self.i

	def r(self, lag):
		"""Get the autocorrelation for a given lag."""
		return self._W[lag - 1] / self._v
