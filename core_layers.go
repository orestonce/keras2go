package keras2go

func K2c_dense(output *K2c_tensor, input *K2c_tensor, kernel *K2c_tensor, bias *K2c_tensor, activation k2c_activationType, fwork []float64) {
	if input.Ndim <= 2 {
		var outrows int
		if input.Ndim > 1 {
			outrows = input.Shape[0]
		} else {
			outrows = 1
		}
		var outcols = kernel.Shape[1]
		var innerdim = kernel.Shape[0]
		var outsize = outrows * outcols
		k2c_affine_matmul(output.Array, input.Array, kernel.Array, bias.Array, outrows, outcols, innerdim)
		activation(output.Array[:outsize])
	} else {
		var axesA = []int{input.Ndim - 1}
		var axesB = []int{0}
		var naxes = 1
		var normalize = 0
		k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork)
		k2c_bias_add(output, bias)
		activation(output.Array[:output.Numel])
	}
}

func K2c_flatten(output *K2c_tensor, input *K2c_tensor) {
	copy(output.Array, input.Array[:input.Numel])
	for i := 0; i < input.Ndim; i++ {
		output.Shape[i] = 1
	}
	output.Shape[0] = input.Numel
	output.Numel = input.Numel
	output.Ndim = 1
}

func K2c_reshape(output *K2c_tensor, input *K2c_tensor, newshp []int) {
	copy(output.Array, input.Array[:input.Numel])
	for i := 0; i < len(newshp); i++ {
		output.Shape[i] = newshp[i]
	}
	output.Ndim = len(newshp)
	output.Numel = input.Numel
}

func K2c_permute_dims(output *K2c_tensor, input *K2c_tensor, permute []int) {
	var Asub, Bsub [K2C_MAX_NDIM]int
	var newshp, oldshp [K2C_MAX_NDIM]int
	var ndim = input.Ndim
	var bidx = 0
	for i := 0; i < ndim; i++ {
		oldshp[i] = input.Shape[i]
	}
	for i := 0; i < ndim; i++ {
		newshp[i] = oldshp[permute[i]]
	}
	for i := 0; i < input.Numel; i++ {
		k2c_idx2sub(i, Asub[:], oldshp[:], ndim)
		for j := 0; j < ndim; j++ {
			Bsub[j] = Asub[permute[j]]
		}
		bidx = k2c_sub2idx(Bsub[:], newshp[:], ndim)
		output.Array[bidx] = input.Array[i]
	}
}

func K2c_repeat_vector(output *K2c_tensor, input *K2c_tensor, n int) {
	var in_width = input.Shape[0]
	for i := 0; i < n; i++ {
		for j := 0; j < in_width; j++ {
			output.Array[i*in_width+j] = input.Array[j]
		}
	}
}
