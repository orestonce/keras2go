package keras2go

/**
* Batch normalization layer.
* applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
*
* :param outputs: output tensor.
* :param inputs: input tensor.
* :param mean: tensor of mean values.
* :param stdev: tensor of standard deviation values.
* :param gamma: tensor of gamma (scale) values.
* :param beta: tensor of beta (offset) values.
* :param axis: axis to be normalized.
*/
func k2c_batch_norm(output *K2c_tensor, input *K2c_tensor, mean *K2c_tensor, stdev *K2c_tensor, gamma *K2c_tensor, beta *K2c_tensor, axis int) {
	var offset = 1
	for i := axis + 1; i < input.Ndim; i++ {
		offset *= input.Shape[i]
	}
	var step = input.Shape[axis]
	for i := 0; i < input.Numel; i++ {
		var idx = (i / offset) % step
		output.Array[i] = (input.Array[i]-mean.Array[idx])/
			stdev.Array[idx]*
			gamma.Array[idx] +
			beta.Array[idx]
	}
}
