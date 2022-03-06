package keras2go

import (
	"math"
)

/**
* Linear activation function.
*   y=x
*
* :param x: Array of input values. Gets overwritten by output.
 */
func K2c_linear(x []float64) {

}

/**
* Exponential activation function.
*   y = exp(x)
*
* :param x: Array of input values. Gets overwritten by output.
 */
func K2c_exponential(x []float64) {
	for idx, value := range x {
		x[idx] = math.Exp(value)
	}
}

/**
* ReLU activation function.
*   y = max(x,0)
*
* :param x: Array of input values. Gets overwritten by output.
 */
func K2c_relu(x []float64) {
	for idx, value := range x {
		if value <= 0 {
			x[idx] = 0
		}
	}
}

/**
* ReLU activation function.
*   y = {1          if      x> 2.5}
*       {0.2*x+0.5  if -2.5<x< 2.5}
*       {0          if      x<-2.5}
*
* :param x: Array of input values. Gets overwritten by output.
 */
func K2c_hard_sigmoid(x []float64) {
	for idx, value := range x {
		if value <= -2.5 {
			x[idx] = 0
		} else if value >= 2.5 {
			x[idx] = 1
		} else {
			x[idx] = 0.2*x[idx] + 0.5
		}
	}
}

/**
 * Tanh activation function.
 *   y = tanh(x)
 *
 * :param x: Array of input values. Gets overwritten by output.
 */
func K2c_tanh(x []float64) {
	for idx, value := range x {
		x[idx] = math.Tanh(value)
	}
}


/**
 * Sigmoid activation function.
 *   y = 1/(1+exp(-x))
 *
 * :param x: Array of input values. Gets overwritten by output.
 */
func K2c_sigmoid(x []float64) {
	for idx, value := range x {
		x[idx] = 1 / (1 + math.Exp(-value))
	}
}

/**
 * Soft max activation function.
 *   z[i] = exp(x[i]-max(x))
 *   y = z/sum(z)
 *
 * :param x: Array of input values. Gets overwritten by output.
 */
func K2c_softmax(x []float64) {
	xmax := x[0]
	var sum float64
	for _, value := range x {
		if value > xmax {
			xmax = value
		}
	}

	for idx, value := range x {
		x[idx] = math.Exp(value - xmax)
	}

	for _, value := range x {
		sum += value
	}

	sum = 1 / sum
	for idx, value := range x {
		x[idx] = value * sum
	}
}


/**
 * Soft plus activation function.
 *   y = ln(1+exp(x))
 *
 * :param x: Array of input values. Gets overwritten by output.
 */
func K2c_softplus(x []float64) {
	for idx, value := range x {
		x[idx] = math.Log1p(math.Exp(value))
	}
}

/**
 * Soft sign activation function.
 *   y = x/(1+|x|)
 *
 * :param x: Array of input values. Gets overwritten by output.
 */
func K2c_softsign(x []float64) {
	for idx, value := range x {
		x[idx] = value / (1 + math.Abs(value))
	}
}

/**
 * Leaky version of a Rectified Linear Unit.
 * It allows a small gradient when the unit is not active:
 *   y = {alpha*x    if x < 0}
 *       {x          if x >= 0}
 *
 * :param x: Array of input values. Gets overwritten by output.
 * :param alpha: slope of negative portion of activation curve.
 */
func k2c_LeakyReLU(x []float64, alpha float64) {
	for idx, value := range x {
		if value < 0 {
			x[idx] = alpha * value
		}
	}
}


/**
 * Parametric Rectified Linear Unit.
 * It allows a small gradient when the unit is not active:
 *   y = {alpha*x    if x < 0}
 *       {x          if x >= 0}
 * Where alpha is a learned Array with the same Shape as x.
 *
 * :param x: Array of input values. Gets overwritten by output.
 * :param alpha: slope of negative portion of activation curve for each unit.
 */
func k2c_PReLU(x []float64, alpha []float64) {
	for idx := range x {
		if x[idx] < 0 {
			x[idx] = x[idx] * alpha[idx]
		}
	}
}


/**
 * Exponential Linear Unit activation (ELU).
 *   y = {alpha*(exp(x) - 1)  if x <  0}
 *       {x                   if x >= 0}
 *
 * :param x: Array of input values. Gets overwritten by output.
 * :param alpha: slope of negative portion of activation curve.
 */
func k2c_ELU(x []float64, alpha float64) {
	for idx, value := range x {
		if value < 0 {
			x[idx] = alpha * math.Expm1(value)
		}
	}
}


/**
 * Thresholded Rectified Linear Unit.
 *   y = {x    if x >  theta}
         {0    if x <= theta}
 *
 * :param x: Array of input values. Gets overwritten by output.
 * :param theta: threshold for activation.
 */
func k2c_ThresholdedReLU(x []float64, theta float64) {
	for idx, value := range x {
		if value < theta {
			x[idx] = 0
		}
	}
}

/**
 * Rectified Linear Unit activation function.
 *   y = {max_value       if          x >= max_value}
 *       {x               if theta <= x <  max_value}
 *       {alpha*(x-theta) if          x < theta}
 *
 * :param x: Array of input values. Gets overwritten by output.
 * :param max_value: maximum value for activated x.
 * :param alpha: slope of negative portion of activation curve.
 * :param theta: threshold for activation.
 */

func k2c_ReLU(x []float64, max_value float64, alpha float64, theta float64) {
	for idx, value := range x {
		if value >= max_value {
			x[idx] = max_value
		} else if value < theta {
			x[idx] = alpha * (value - theta)
		}
	}
}
