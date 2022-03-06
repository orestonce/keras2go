package keras2go

/**
* Rank of largest keras2c tensors.
* mostly used to ensure a standard size for the tensor.Shape Array.
 */
const K2C_MAX_NDIM = 5

/**
* tensor type for keras2c.
 */
type K2c_tensor struct {
	Array []float64         /** Pointer to Array of tensor values flattened in row major order. */
	Ndim  int               /** Rank of the tensor (number of dimensions). */
	Numel int               /** Number of elements in the tensor. */
	Shape [K2C_MAX_NDIM]int /** Array, size of the tensor in each dimension. */
}

type k2c_activationType func(x []float64)
