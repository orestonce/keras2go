package keras2go

/**
* Embedding Layer.
* turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
*
* :param output: output tensor.
* :param input: input tensor.
* :param kernel: kernel mapping integers to vectors.
*/
func k2c_embedding(output *K2c_tensor, input *K2c_tensor, kernel *K2c_tensor) {
	var output_dim = kernel.Shape[1]
	for i := 0; i < input.Numel; i++ {
		for j := 0; j < output_dim; j++ {
			output.Array[i*output_dim+j] = kernel.Array[int(input.Array[i])*output_dim+j]
		}
	}
}
