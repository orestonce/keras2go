package keras2go

/**
* Element-wise sum of several tensors.
*
* :param output: output tensor.
* :param num_tensors: number of tensors being summed.
* :param ...: variadic. Tensors to be summed.
*/
func k2c_add(output *K2c_tensor, inputList ...*K2c_tensor) {
	output.fillFloat64(0)
	for _, input := range inputList {
		for j := 0; j < output.Numel; j++ {
			output.Array[j] += input.Array[j]
		}
	}
}


/**
* Element-wise difference of two tensors.
*
* :param output: output tensor.
* :param num_tensors: number of tensors being summed. Not used but kept for a consistent API with other merge layers.
* :param tensor1: first input tensor.
* :param tensor2: second input tensor.
*/
func k2c_subtract(output *K2c_tensor, num_tensors int, tensor1 *K2c_tensor, tensor2 *K2c_tensor) {
	for i := 0; i < output.Numel; i++ {
		output.Array[i] = tensor1.Array[i] - tensor2.Array[i]
	}
}


/**
* Element-wise product of several tensors.
*
* :param output: output tensor.
* :param num_tensors: number of tensors being multiplied.
* :param ...: variadic. Tensors to be multiplied.
*/
func k2c_multiply(output *K2c_tensor, inputList ...*K2c_tensor) {
	output.fillFloat64(1)
	for _, input := range inputList {
		for j := 0; j < output.Numel; j++ {
			output.Array[j] *= input.Array[j]
		}
	}
}


/**
* Element-wise average of several tensors.
*
* :param output: output tensor.
* :param num_tensors: number of tensors being averaged.
* :param ...: variadic. Tensors to be averaged.
*/
func k2c_average(output *K2c_tensor, inputList ...*K2c_tensor) {
	var num_tensors_inv = 1.0 / float64(len(inputList))
	output.fillFloat64(0)
	for _, input := range inputList {
		for j := 0; j < output.Numel; j++ {
			output.Array[j] += input.Array[j] * num_tensors_inv
		}
	}
}


/**
* Element-wise maximum of several tensors.
*
* :param output: output tensor.
* :param num_tensors: number of tensors over which to take max.
* :param ...: variadic. Tensors to take the max of.
*/
func k2c_max(output *K2c_tensor, inputList ...*K2c_tensor) {
	for i := 0; i < output.Numel; i++ {
		output.Array[i] = inputList[0].Array[i]
	}
	for _, input := range inputList[1:] {
		for j := 0; j < output.Numel; j++ {
			if output.Array[j] < input.Array[j] {
				output.Array[j] = input.Array[j]
			}
		}
	}
}


/**
* Element-wise minimum of several tensors.
*
* :param output: output tensor.
* :param inputList: Tensors to take the min of.
*/
func k2c_min(output *K2c_tensor, inputList ...*K2c_tensor) {
	for i := 0; i < output.Numel; i++ {
		output.Array[i] = inputList[0].Array[i]
	}
	for _, input := range inputList[1:] {
		for j := 0; j < output.Numel; j++ {
			if output.Array[j] > input.Array[j] {
				output.Array[j] = input.Array[j]
			}
		}
	}
}


/**
* Concatenation of several tensors.
*
* :param output: output tensor.
* :param axis: axis along which to concatenate.
* :param inputList: Tensors to concatenate.
*/
func k2c_concatenate(output *K2c_tensor, axis int, inputList ...*K2c_tensor) {
	var offset = 0
	var outidx int
	var insub, outsub [K2C_MAX_NDIM]int
	for _, input := range inputList {
		for j := 0; j < input.Numel; j++ {
			k2c_idx2sub(j, insub[:], input.Shape[:], input.Ndim)
			copy(outsub[:], insub[:])
			outidx = k2c_sub2idx(outsub[:], output.Shape[:], output.Ndim)
			output.Array[outidx] = input.Array[j]
		}
		offset += input.Shape[axis]
	}
}
