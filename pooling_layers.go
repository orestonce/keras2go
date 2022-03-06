package keras2go

import "math"

func K2c_global_max_pooling(output *K2c_tensor, input *K2c_tensor) {
	var in_chan = input.Shape[input.Ndim-1]
	copy(output.Array, input.Array[:in_chan])

	for i := 0; i < input.Numel; i += in_chan {
		for j := 0; j < in_chan; j++ {
			if output.Array[j] < input.Array[i+j] {
				output.Array[j] = input.Array[i+j]
			}
		}
	}
}

func K2c_global_avg_pooling(output *K2c_tensor, input *K2c_tensor) {
	var in_chan = input.Shape[input.Ndim-1]
	output.fillFloat64(0)
	num_inv := 1 / float64(input.Numel/in_chan)

	for i := 0; i < input.Numel; i += in_chan {
		for j := 0; j < in_chan; j++ {
			output.Array[j] += input.Array[i+j] * num_inv
		}
	}
}

func K2c_maxpool1d(output *K2c_tensor, input *K2c_tensor, pool_size int, stride int) {
	var channels = input.Shape[1]

	for i := 0; i < channels; i++ {
		var j, k int
		for j < output.Shape[0]*channels {
			output.Array[j+i] = input.Array[k+i]
			for l := 0; l < pool_size*channels; l += channels {
				if output.Array[j+i] < input.Array[k+i+l] {
					output.Array[j+i] = input.Array[k+i+l]
				}
			}
			j += channels
			k += stride * channels
		}
	}
}

func K2c_maxpool2d(output *K2c_tensor, input *K2c_tensor, pool_size []int, stride []int) {
	var channels = input.Shape[2]
	for i := 0; i < channels; i++ {
		var j, k int
		for j < output.Shape[1]*channels {
			var l, m int
			for l < output.Numel {
				output.Array[l+j+i] = input.Array[m+k+i]
				for n := 0; n < pool_size[1]*channels; n += channels {
					for p := 0; p < pool_size[0]*channels*input.Shape[1]; p += channels * input.Shape[1] {
						if output.Array[l+j+i] < input.Array[m+k+i+n+p] {
							output.Array[l+j+i] = input.Array[m+k+i+n+p]
						}
					}
				}
				l += channels * output.Shape[1]
				m += channels * input.Shape[1] * stride[0]
			}

			j += channels
			k += channels * stride[1]
		}
	}
}


func K2c_avgpool1d(output *K2c_tensor, input *K2c_tensor, pool_size int, stride int) {
	var channels = input.Shape[1]
	output.fillFloat64(0)
	for i := 0; i < channels; i++ {
		var j, k int
		for j < output.Numel {
			var count int
			for l := 0; l < pool_size*channels; l += channels {
				if input.Array[k+i+l] > -math.MaxFloat64 {
					output.Array[j+i] += input.Array[k+i+l]
					count++
				}
			}
			output.Array[i+j] /= float64(count)
			j += channels
			k += stride * channels
		}
	}
}

func K2c_avgpool2d(output *K2c_tensor, input *K2c_tensor, pool_size []int, stride []int) {
	output.fillFloat64(0)
	var channels = input.Shape[2]
	for i := 0; i < channels; i++ {
		var j, k int
		for j < output.Shape[1]*channels {
			var l, m int
			for l < output.Numel {
				var count int
				for n := 0; n < pool_size[1]*channels; n += channels {
					for p := 0; p < pool_size[0]*channels*input.Shape[1]; p += channels * input.Shape[1] {
						if -math.MaxFloat64 < input.Array[m+k+i+n+p] {
							output.Array[l+j+i] += input.Array[m+k+i+n+p]
							count++
						}
					}
				}
				output.Array[l+j+i] /= float64(count)
				l += channels * output.Shape[1]
				m += channels * input.Shape[1] * stride[0]
			}
			j += channels
			k += channels * stride[1]
		}
	}
}
