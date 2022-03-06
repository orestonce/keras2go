package keras2go

func k2c_lstmcell(state []float64, input []float64, kernel *K2c_tensor, recurrent_kernel *K2c_tensor, bias *K2c_tensor, fwork []float64, recurrent_activation k2c_activationType, output_activation k2c_activationType) {
	var units = recurrent_kernel.Shape[1]
	var in_width = kernel.Shape[0] / 4

	var h_tm1 = state
	var c_tm1 = state[units:]
	var outrows = 1
	var Wi = kernel.Array
	var Wf = kernel.Array[in_width*units:]
	var Wc = kernel.Array[2*in_width*units:]
	var Wo = kernel.Array[3*in_width*units:]
	var Ui = recurrent_kernel.Array
	var Uf = recurrent_kernel.Array[units*units:]
	var Uc = recurrent_kernel.Array[2*units*units:]
	var Uo = recurrent_kernel.Array[3*units*units:]
	var bi = bias.Array
	var bf = bias.Array[units:]
	var bc = bias.Array[2*units:]
	var bo = bias.Array[3*units:]
	var xi = fwork
	var xf = fwork[units:]
	var xc = fwork[2*units:]
	var xo = fwork[3*units:]
	var yi = fwork[4*units:]
	var yf = fwork[5*units:]
	var yc = fwork[6*units:]
	var yo = fwork[7*units:]

	k2c_affine_matmul(xi, input, Wi, bi, outrows, units, in_width)
	//xi = input*Wi + bi;
	k2c_affine_matmul(xi, input, Wi, bi, outrows, units, in_width)
	//xf = input*Wf + bf;
	k2c_affine_matmul(xf, input, Wf, bf, outrows, units, in_width)
	//xc = input*Wc + bc;
	k2c_affine_matmul(xc, input, Wc, bc, outrows, units, in_width)
	//xo = input*Wo + bo;
	k2c_affine_matmul(xo, input, Wo, bo, outrows, units, in_width)

	// yi = recurrent_activation(xi + h_tm1*Ui);
	k2c_affine_matmul(yi, h_tm1, Ui, xi, outrows, units, units)
	recurrent_activation(yi[:units])

	// yf = recurrent_activation(xf + h_tm1*Uf);
	k2c_affine_matmul(yf, h_tm1, Uf, xf, outrows, units, units)
	recurrent_activation(yf[:units])

	// yc = yf.*c_tm1 + yi.*output_activation(xc + h_tm1*Uc);
	k2c_affine_matmul(yc, h_tm1, Uc, xc, outrows, units, units)
	output_activation(yc[:units])
	for i := 0; i < units; i++ {
		yc[i] = yf[i]*c_tm1[i] + yi[i]*yc[i]
	}

	// yo = recurrent_activation(xo + h_tm1*Uo);
	k2c_affine_matmul(yo, h_tm1, Uo, xo, outrows, units, units)
	recurrent_activation(yo[:units])

	// h = yo.*output_activation(yc);
	// state = [h;yc];
	for i := 0; i < units; i++ {
		state[units+i] = yc[i]
	}

	output_activation(yc[:units])

	for i := 0; i < units; i++ {
		state[i] = yo[i] * yc[i]
	}
}

func K2c_lstm(output *K2c_tensor, input *K2c_tensor, state []float64, kernel *K2c_tensor, recurrent_kernel *K2c_tensor, bias *K2c_tensor, fwork []float64, go_backwards int, return_sequences int, recurrent_activation k2c_activationType, output_activation k2c_activationType) {
	var in_height = input.Shape[0]
	var in_width = input.Shape[1]
	var units = recurrent_kernel.Shape[1]

	if go_backwards != 0 {
		for i := in_height - 1; i > -1; i-- {
			k2c_lstmcell(state, input.Array[i*in_width:], kernel, recurrent_kernel,
				bias, fwork, recurrent_activation, output_activation)
			if return_sequences != 0 {
				for j := 0; j < units; j++ {
					output.Array[(in_height-1-i)*units+j] = state[j]
				}
			}
		}
	} else {
		for i := 0; i < in_height; i++ {
			k2c_lstmcell(state, input.Array[i*in_width:], kernel, recurrent_kernel,
				bias, fwork, recurrent_activation, output_activation)
			if return_sequences != 0 {
				for j := 0; j < units; j++ {
					output.Array[i*units+j] = state[j]
				}
			}
		}
	}
	if return_sequences == 0 {
		for i := 0; i < units; i++ {
			output.Array[i] = state[i]
		}
	}
}


/**
* Cell for the RNN layer.
* "units" is the dimension of the output space
*
* :param state: Array[units] recurrent state.
* :param input: Array of input data.
* :param kernel: kernel tensor.
* :param recurrent_kernel: recurrent kernel tensor
* :param bias: bias tensor.
* :param fwork: Array[2*units] working storage.
* :param output_activation: activation function to apply to output.
*/
func k2c_simpleRNNcell(state []float64, input []float64, kernel *K2c_tensor, recurrent_kernel *K2c_tensor, bias *K2c_tensor, fwork []float64, output_activation k2c_activationType) {
	var units = recurrent_kernel.Shape[1]
	var in_width = kernel.Shape[0]

	var outrows = 1
	var h1 = fwork
	var h2 = fwork[units:]
	// h1 = input*kernel+bias
	k2c_affine_matmul(h1, input, kernel.Array, bias.Array, outrows, units, in_width)

	// h2 = state*recurrent_kernel + h1
	k2c_affine_matmul(h2, state, recurrent_kernel.Array, h1, outrows, units, units)
	output_activation(h2[:units])

	for i := 0; i < units; i++ {
		state[i] = h2[i]
	}
}

/**
* Fully-connected RNN where the output is to be fed back to input.
* "units" is the dimension of the output space
*
* :param output: output tensor.
* :param input: input tensor.
* :param state: Array[units] recurrent state.
* :param kernel: kernel tensor.
* :param recurrent_kernel: recurrent kernel tensor
* :param bias: bias tensor.
* :param fwork: Array[2*units] working storage.
* :param go_backwards: whether to process input sequences forwards (1) or backwards (0).
* :param return_sequences: whether to return the last output in the output sequence (0), or the full sequence (1).
* :param output_activation: activation function to apply to output.
*/
func k2c_simpleRNN(output *K2c_tensor, input *K2c_tensor, state []float64, kernel *K2c_tensor, recurrent_kernel *K2c_tensor, bias *K2c_tensor, fwork []float64, go_backwards int, return_sequences int, output_activation k2c_activationType) {
	var in_width = input.Shape[1]
	var in_height = input.Shape[0]
	var units = recurrent_kernel.Shape[1]

	if go_backwards != 0 {
		for i := in_height - 1; i > -1; i-- {
			k2c_simpleRNNcell(state, input.Array[i*in_width:], kernel, recurrent_kernel, bias, fwork, output_activation)
			if return_sequences != 0 {
				for j := 0; j < units; j++ {
					output.Array[(in_height-1-i)*units+j] = state[j]
				}
			}
		}
	} else {
		for i := 0; i < in_height; i++ {
			k2c_simpleRNNcell(state, input.Array[i*in_width:], kernel, recurrent_kernel, bias, fwork, output_activation)
			if return_sequences != 0 {
				for j := 0; j < units; j++ {
					output.Array[i*units+j] = state[j]
				}
			}
		}
	}
	if return_sequences == 0 {
		for i := 0; i < units; i++ {
			output.Array[i] = state[i]
		}
	}
}


/**
* Cell for the GRU layer.
* "units" is the dimension of the output space
*
* :param state: Array[units] recurrent state.
* :param input: Array of input data.
* :param kernel: kernel tensor.
* :param recurrent_kernel: recurrent kernel tensor
* :param bias: bias tensor.
* :param fwork: Array[6*units] working storage.
* :param reset_after: whether to apply the reset gate before (0) or after (1) the matrix multiplication.
* :param recurrent_activation: activation function to apply to internal state.
* :param output_activation: activation function to apply to output.
*/
func k2c_grucell(state []float64, input []float64, kernel *K2c_tensor, recurrent_kernel *K2c_tensor, bias *K2c_tensor, fwork []float64, reset_after int, recurrent_activation k2c_activationType, output_activation k2c_activationType) {
	var units = recurrent_kernel.Shape[1]
	var in_width = kernel.Shape[0] / 3

	var h_tm1 = state
	var outrows = 1
	var Wz = kernel.Array
	var Wr = kernel.Array[in_width*units:]
	var Wh = kernel.Array[2*in_width*units:]
	var Uz = recurrent_kernel.Array
	var Ur = recurrent_kernel.Array[units*units:]
	var Uh = recurrent_kernel.Array[2*units*units:]
	var bz = bias.Array
	var br = bias.Array[units:]
	var bh = bias.Array[2*units:]
	var rbz = bias.Array[3*units:]
	var rbr = bias.Array[4*units:]
	var rbh = bias.Array[5*units:]
	var xz = fwork
	var xr = fwork[units:]
	var xh = fwork[2*units:]
	var yz = fwork[3*units:]
	var yr = fwork[4*units:]
	var yh = fwork[5*units:]

	//     x_z = input*kernel_z + input_bias_z
	k2c_affine_matmul(xz, input, Wz, bz, outrows, units, in_width)
	//    x_r = input@kernel_r + input_bias_r
	k2c_affine_matmul(xr, input, Wr, br, outrows, units, in_width)
	//    x_h = input@kernel_h + input_bias_h
	k2c_affine_matmul(xh, input, Wh, bh, outrows, units, in_width)

	//   recurrent_z = h_tm1@recurrent_kernel_z
	k2c_affine_matmul(yz, h_tm1, Uz, rbz, outrows, units, units)
	//    recurrent_r = h_tm1@recurrent_kernel_r
	k2c_affine_matmul(yr, h_tm1, Ur, rbr, outrows, units, units)

	//    z = np.tanh(x_z + recurrent_z)
	//    r = np.tanh(x_r + recurrent_r)
	for i := 0; i < units; i++ {
		yz[i] = xz[i] + yz[i]
		yr[i] = xr[i] + yr[i]
	}
	recurrent_activation(yz[:units])
	recurrent_activation(yr[:units])

	//    reset gate applied after/before matrix multiplication
	if reset_after != 0 {
		//        recurrent_h = h_tm1*recurrent_kernel_h + recurrent_bias_h
		k2c_affine_matmul(yh, h_tm1, Uh, rbh, outrows, units, units)
		//        recurrent_h = r .* recurrent_h
		for i := 0; i < units; i++ {
			yh[i] = yr[i] * yh[i]
		}
	} else {
		//        recurrent_h = (r .* h_tm1)*recurrent_kernel_h
		for i := 0; i < units; i++ {
			yh[i] = yr[i] * h_tm1[i]
		}
		k2c_matmul(xz, yh, Uh, outrows, units, units) //reuse xz as new yh
		for i := 0; i < units; i++ {
			yh[i] = xz[i]
		}
	}
	//    hh = np.tanh(x_h + recurrent_h)
	for i := 0; i < units; i++ {
		xr[i] = xh[i] + yh[i] // reuse xr = hh
	}
	output_activation(xr[:units])
	//    h = z .* h_tm1 + (1 - z) .* hh
	for i := 0; i < units; i++ {
		state[i] = yz[i]*h_tm1[i] + (1.0-yz[i])*xr[i]
	}
}

func K2c_gru(output *K2c_tensor, input *K2c_tensor, state []float64, kernel *K2c_tensor, recurrent_kernel *K2c_tensor, bias *K2c_tensor, fwork []float64, reset_after int, go_backwards int, return_sequences int, recurrent_activation k2c_activationType, output_activation k2c_activationType) {
	var in_width = input.Shape[1]
	var in_height = input.Shape[0]
	var units = recurrent_kernel.Shape[1]

	if go_backwards != 0 {
		for i := in_height - 1; i > -1; i-- {
			k2c_grucell(state, input.Array[i*in_width:], kernel, recurrent_kernel, bias, fwork, reset_after, recurrent_activation, output_activation)
			if return_sequences != 0 {
				for j := 0; j < units; j++ {
					output.Array[(in_height-1-i)*units+j] = state[j]
				}
			}
		}
	} else {
		for i := 0; i < in_height; i++ {
			k2c_grucell(state, input.Array[i*in_width:], kernel, recurrent_kernel, bias, fwork, reset_after, recurrent_activation, output_activation)
			if return_sequences != 0 {
				for j := 0; j < units; j++ {
					output.Array[i*units+j] = state[j]
				}
			}
		}
	}
	if return_sequences == 0 {
		for i := 0; i < units; i++ {
			output.Array[i] = state[i]
		}
	}
}
