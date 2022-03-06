package keras2go

import "math"

/**
* 1D (temporal) Padding.
*
* :param output: tensor to store padded output data.
* :param input: tensor to pad.
* :param fill: value to fill in padded areas.
* :param pad: Array[2] of how many rows to pad. Order is {before dim 1, after dim 1}.
 */
func k2c_pad1d(output *K2c_tensor, input *K2c_tensor, fill float64, pad []int) {
	in_width := input.Shape[1]
	pad_top := pad[0]

	output.fillFloat64(fill)

	offset := pad_top * in_width
	copy(output.Array[offset:], input.Array[:input.Numel])
}

func (this *K2c_tensor) fillFloat64(fill float64) {
	if math.Abs(fill) < 1e-6 {
		for idx := 0; idx < this.Numel; idx++ {
			this.Array[idx] = 0
		}
	} else {
		for idx := 0; idx < this.Numel; idx++ {
			this.Array[idx] = fill
		}
	}
}

/**
* 2D (spatial) Padding.
*
* :param output: tensor to store padded output data.
* :param input: tensor to pad.
* :param fill: value to fill in padded areas.
* :param pad: Array[4] of how many rows/cols to pad. Order is {before dim 1, after dim 1, before dim 2, after dim 2}.
 */
func k2c_pad2d(output *K2c_tensor, input *K2c_tensor, fill float64, pad []int) {
	in_height := input.Shape[0]
	in_width := input.Shape[1]
	in_channels := input.Shape[2]
	pad_top := pad[0]
	pad_left := pad[2]
	pad_right := pad[3]

	output.fillFloat64(fill)

	offset := in_channels*(pad_left+pad_right+in_width)*pad_top + in_channels*pad_left
	num := in_channels * in_width
	step := num + in_channels*(pad_left+pad_right)
	for idx := 0; idx < in_height; idx++ {
		copy(output.Array[offset:], input.Array[idx*num:idx*num+num])
		offset += step
	}
}

/**
* 3D (spatial or spatio-temporal) Padding.
*
* :param output: tensor to store padded output data.
* :param input: tensor to pad.
* :param fill: value to fill in padded areas.
* :param pad: Array[6] of how many rows/cols to pad. Order is {before dim 1, after dim 1, before dim 2, after dim 2, before dim 3, after dim 3}.
 */
func k2c_pad3d(output *K2c_tensor, input *K2c_tensor, fill float64, pad []int) {
	dim1 := input.Shape[0]
	dim2 := input.Shape[1]
	dim3 := input.Shape[2]
	//outdim1 := dim1 + pad[0] + pad[1]
	outdim2 := dim2 + pad[2] + pad[3]
	outdim3 := dim3 + pad[4] + pad[5]
	in_channels := input.Shape[3]

	output.fillFloat64(fill)

	offset1 := in_channels*(outdim2*outdim3)*pad[0] + in_channels*outdim3*pad[2] + in_channels*pad[4]
	num := in_channels * dim3
	outstep2 := num + in_channels*(pad[4]+pad[5])
	outstep1 := outdim2 * outdim3 * in_channels
	instep1 := dim2 * dim3 * in_channels
	instep2 := dim3 * in_channels

	for i := 0; i < dim1; i++ {
		for j := 0; j < dim2; j++ {
			inIdx := i*instep1 + j*instep2
			copy(output.Array[offset1+i*outstep1+j*outstep2:], input.Array[inIdx:inIdx+num])
		}
	}
}

/**
* 1D (temporal) Convolution.
* Assumes a "channels last" structure.
*
* :param output: output tensor.
* :param input: input tensor.
* :param kernel: kernel tensor.
* :param bias: bias tensor.
* :param stride: stride length of the convolution.
* :param dilation: dilation rate to use for dilated convolution.
* :param activation: activation function to apply to output.
 */
func k2c_conv1d(output *K2c_tensor, input *K2c_tensor, kernel *K2c_tensor, bias *K2c_tensor, stride int, dilation int, activation k2c_activationType) {
	output.fillFloat64(0)

	out_times := output.Shape[0]
	out_channels := output.Shape[1]
	in_channels := input.Shape[1]

	for x0 := 0; x0 < out_times; x0++ {
		for z := 0; z < kernel.Shape[0]; z++ {
			for q := 0; q < in_channels; q++ {
				for k := 0; k < out_channels; k++ {
					output.Array[x0*out_channels+k] += kernel.Array[z*(kernel.Shape[2]*kernel.Shape[1])+q*(kernel.Shape[2])+k] * input.Array[(x0*stride+dilation*z)*in_channels+q]
				}
			}
		}
	}
	k2c_bias_add(output, bias)
	activation(output.Array[:output.Numel])
}


/**
* 2D (spatial) Convolution.
* Assumes a "channels last" structure.
*
* :param output: output tensor.
* :param input: input tensor.
* :param kernel: kernel tensor.
* :param bias: bias tensor.
* :param stride: Array[2] of stride length of the convolution. Order is {stride dim 1, stride dim 2}.
* :param dilation: Array[2] dilation rate to use for dilated convolution. Order is {dilation dim 1, dilation dim 2}.
* :param activation: activation function to apply to output.
*/
func k2c_conv2d(output *K2c_tensor, input *K2c_tensor, kernel *K2c_tensor, bias *K2c_tensor, stride []int, dilation []int, activation k2c_activationType) {
	output.fillFloat64(0)
	out_rows := output.Shape[0]
	out_cols := output.Shape[1]
	out_channels := output.Shape[2]
	in_channels := input.Shape[2]

	for x0 := 0; x0 < out_rows; x0++ {
		for x1 := 0; x1 < out_cols; x1++ {
			for z0 := 0; z0 < kernel.Shape[0]; z0++ {
				for z1 := 0; z1 < kernel.Shape[1]; z1++ {
					for q := 0; q < in_channels; q++ {
						for k := 0; k < out_channels; k++ {
							output.Array[x0*(output.Shape[2]*output.Shape[1])+
								x1*(output.Shape[2])+k] +=
								kernel.Array[z0*(kernel.Shape[3]*kernel.Shape[2]*kernel.Shape[1])+
									z1*(kernel.Shape[3]*kernel.Shape[2])+
									q*(kernel.Shape[3]+k)] *
									input.Array[(x0+stride[0]+dilation[0]*z0)*
										(input.Shape[2]*input.Shape[1])+
										(x1*stride[1]+dilation[1]*z1)*(input.Shape[2])+q]
						}
					}
				}
			}
		}
	}
	k2c_bias_add(output, bias)
	activation(output.Array[:output.Numel])
}


/**
* 3D (spatial or spatio-temporal) Convolution.
* Assumes a "channels last" structure.
*
* :param output: output tensor.
* :param input: input tensor.
* :param kernel: kernel tensor.
* :param bias: bias tensor.
* :param stride: Array[3] of stride length of the convolution. Order is {stride dim 1, stride dim 2, stride dim 3}.
* :param dilation: Array[3] dilation rate to use for dilated convolution. Order is {dilation dim 1, dilation dim 2, dilation dim 3}.
* :param activation: activation function to apply to output.
*/
func k2c_conv3d(output *K2c_tensor, input *K2c_tensor, kernel *K2c_tensor, bias *K2c_tensor, stride []int, dilation []int, activation k2c_activationType) {
	output.fillFloat64(0)
	dim1 := output.Shape[0]
	dim2 := output.Shape[1]
	dim3 := output.Shape[2]
	out_channels := output.Shape[3]
	in_channels := input.Shape[3]

	for x0 := 0; x0 < dim1; x0++ {
		for x1 := 0; x1 < dim2; x1++ {
			for x2 := 0; x2 < dim3; x2++ {
				for z0 := 0; z0 < kernel.Shape[0]; z0++ {
					for z1 := 0; z1 < kernel.Shape[1]; z1++ {
						for z2 := 0; z2 < kernel.Shape[2]; z2++ {
							for q := 0; q < in_channels; q++ {
								for k := 0; k < out_channels; k++ {
									output.Array[x0*(output.Shape[3]*output.Shape[2]*
										output.Shape[1])+
										x1*(output.Shape[3]*output.Shape[2])+
										x2*(output.Shape[3])+k] +=
										kernel.Array[z0*(kernel.Shape[4]*kernel.Shape[3]*kernel.Shape[2]*kernel.Shape[1])+
											z1*(kernel.Shape[4]*kernel.Shape[3]*kernel.Shape[2])+
											z2*(kernel.Shape[4]*kernel.Shape[3])+
											q*(kernel.Shape[4])+k] *
											input.Array[(x0*stride[0]+dilation[0]*z0)*
												(input.Shape[3]*input.Shape[2]*input.Shape[1])+
												(x1*stride[1]+dilation[1]*z1)*(input.Shape[3]*input.Shape[2])+
												(x2*stride[2]+dilation[2]*z2)*(input.Shape[3])+q]
								}
							}
						}
					}
				}
			}
		}
	}

	k2c_bias_add(output, bias)
	activation(output.Array[:output.Numel])
}


/**
* 1D (temporal) Cropping.
*
* :param output: tensor to store cropped output data.
* :param input: tensor to crop.
* :param pad: Array[2] of how many rows to crop. Order is {before dim 1, after dim 1}.
*/
func k2c_crop1d(output *K2c_tensor, input *K2c_tensor, crop []int) {
	offset := crop[0] * input.Shape[1]
	copy(output.Array, input.Array[offset:offset+output.Numel])
}


/**
* 2D (spatial) Cropping.
*
* :param output: tensor to store cropped output data.
* :param input: tensor to crop.
* :param pad: Array[4] of how many rows/cols to crop. Order is {before dim 1, after dim 1, before dim 2, after dim 2}.
*/
func k2c_crop2d(output *K2c_tensor, input *K2c_tensor, crop []int) {
	var out_height = output.Shape[0]
	var in_width = input.Shape[1]
	var in_channels = input.Shape[2]
	var crop_top = crop[0]
	var crop_left = crop[2]
	var crop_right = crop[3]

	var offset = in_channels*in_width*crop_top + in_channels*crop_left
	var num = in_channels * (in_width - crop_left - crop_right)
	for i := 0; i < out_height; i++ {
		copy(output.Array[i*num:], input.Array[offset:offset+num])
		offset += in_width * in_channels
	}
}


/**
* 3D (spatial or spatio-temporal) Cropping.
*
* :param output: tensor to store cropped output data.
* :param input: tensor to crop.
* :param pad: Array[6] of how many rows/cols to crop. Order is {before dim 1, after dim 1, before dim 2, after dim 2, before dim 3, after dim 3}.
*/
func k2c_crop3d(output *K2c_tensor, input *K2c_tensor, crop []int) {
	var dim1 = input.Shape[0]
	var dim2 = input.Shape[1]
	var dim3 = input.Shape[2]
	var outdim1 = dim1 - crop[0] - crop[1]
	var outdim2 = dim2 - crop[2] - crop[3]
	var outdim3 = dim3 - crop[4] - crop[5]
	var in_channels = input.Shape[3]

	var offset1 = in_channels*(dim2*dim3)*crop[0] + in_channels*dim3*crop[2] + in_channels*crop[4]
	var num = in_channels * outdim3
	var instep2 = num + in_channels*(crop[4]+crop[5])
	var instep1 = dim2 * dim3 * in_channels
	var outstep1 = outdim2 * outdim3 * in_channels
	var outstep2 = outdim3 * in_channels

	for i := 0; i < outdim1; i++ {
		for j := 0; j < outdim2; j++ {
			inIdx := offset1 + i*instep1 + j*instep2
			copy(output.Array[i*outstep1+j*outstep2:], input.Array[inIdx:inIdx+num])
		}
	}
}


/**
* 1D (temporal) Upsampling.
* Repeats each temporal step size times along the time axis.
*
* :param output: output tensor.
* :param input: input tensor.
* :param size: Upsampling factor.
*/
func k2c_upsampling1d(output *K2c_tensor, input *K2c_tensor, size int) {
	var in_height = input.Shape[0]
	var in_width = input.Shape[1]

	for i := 0; i < in_height; i++ {
		for j := 0; j < size; j++ {
			for k := 0; k < in_width; k++ {
				output.Array[(size*i+j)*in_width+k] = input.Array[i*in_width+k]
			}
		}
	}
}


/**
* 2D (spatial) Upsampling.
* Repeats the rows and columns of the data by size[0] and size[1] respectively.
*
* :param output: output tensor.
* :param input: input tensor.
* :param size: Array[2] of upsampling factors. Order is {upsampling dim 1, upsampling dim 2}.
*/
func k2c_upsampling2d(output *K2c_tensor, input *K2c_tensor, size []int) {
	var out_height = output.Shape[0]
	var out_width = output.Shape[1]
	var channels = output.Shape[2]

	for i := 0; i < out_height; i++ {
		for j := 0; j < out_width; j++ {
			var insub = [K2C_MAX_NDIM]int{i / size[0], j / size[1], 0}
			var outsub = [K2C_MAX_NDIM]int{i, j, 0}

			inIdx := k2c_sub2idx(insub[:], input.Shape[:], input.Ndim)
			copy(output.Array[k2c_sub2idx(outsub[:], output.Shape[:], output.Ndim):], input.Array[inIdx:inIdx+channels])
		}
	}
}


/**
* 2D (spatial) Upsampling.
* Repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1] and size[2] respectively.
*
* :param output: output tensor.
* :param input: input tensor.
* :param size: Array[3] of upsampling factors. Order is {upsampling dim 1, upsampling dim 2, upsampling dim 3}.
*/
func k2c_upsampling3d(output *K2c_tensor, input *K2c_tensor, size []int) {
	var dim1 = output.Shape[0]
	var dim2 = output.Shape[1]
	var dim3 = output.Shape[2]
	var channels = input.Shape[3]

	for i := 0; i < dim1; i++ {
		for j := 0; j < dim2; j++ {
			for k := 0; k < dim3; k++ {
				var insub = [K2C_MAX_NDIM]int{i / size[0], j / size[1], k / size[2], 0}
				var outsub = [K2C_MAX_NDIM]int{i, j, k, 0}
				inIdx := k2c_sub2idx(insub[:], input.Shape[:], input.Ndim)
				copy(output.Array[k2c_sub2idx(outsub[:], output.Shape[:], output.Ndim):], input.Array[inIdx:inIdx+channels])
			}
		}
	}
}
