package keras2go

import "math"

/**
* Just your basic 1d matrix multipication.
* computes C = A*B
* assumes A,B,C are all 1d arrays of matrices stored in row major order.
*
* :param C: output Array.
* :param A: input Array 1.
* :param B: input Array 2.
* :param outrows: number of rows of C and A.
* :param outcols: number of cols of C and B.
* :param innderdim: number of cols of A and rows of B
*/
func k2c_matmul(C []float64, A []float64, B []float64, outrows int, outcols int, innerdim int) {
	float64SliceToZero(C)

	for i := 0; i < outrows; i++ {
		var outrowidx = i * outcols
		var inneridx = i * innerdim
		for k := 0; k < innerdim; k++ {
			for j := 0; j < outcols; j++ {
				C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j]
			}
		}
	}
}

func float64SliceToZero(a []float64) {
	for idx := 0; idx < len(a); idx++ {
		a[idx] = 0
	}
}

/**
* Affine matrix multiplication.
* computes C = A*B + d, where d is a vector that is added to each
row of A*B
* assumes A,B,C are all 1d arrays of matrices stored in row major order
*
* :param C: output Array.
* :param A: input Array 1.
* :param B: input Array 2.
* :param d: input Array 3.
* :param outrows: number of rows of C and A.
* :param outcols: number of cols of C, B and d.
* :param innderdim: number of cols of A and rows of B
*/
func k2c_affine_matmul(C []float64, A []float64, B []float64, d []float64, outrows int, outcols int, innerdim int) {
	float64SliceToZero(C)

	for i := 0; i < outrows; i++ {
		var outrowidx = i * outcols
		var inneridx = i * innerdim
		for j := 0; j < outcols; j++ {
			for k := 0; k < innerdim; k++ {
				C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j]
			}
			C[outrowidx+j] += d[j]
		}
	}
}


/**
* Converts subscripts to linear indices in row major order.
*
* :param sub: Array[Ndim] subscript to convert.
* :param Shape: Array[Ndim] Shape of Array being indexed.
* :param Ndim: number of dimensions of Array being indexed.
* :return: linear index in row major order.
*/
func k2c_sub2idx(sub []int, shape []int, ndim int) int {
	var idx = 0
	var temp = 0
	for i := 0; i < ndim; i++ {
		temp = sub[i]
		for j := ndim - 1; j > i; j-- {
			temp *= shape[j]
		}
		idx += temp
	}
	return idx
}


/**
* Converts linear indices to subscripts in row major order.
*
* :param idx: linear index in row major order.
* :param sub: Array[Ndim] output subscript.
* :param Shape: Array[Ndim] Shape of Array being indexed.
* :param Ndim: number of dimensions of Array being indexed.
*/
func k2c_idx2sub(idx int, sub []int, shape []int, ndim int) {
	idx2 := idx
	for i := ndim - 1; i >= 0; i-- {
		sub[i] = idx2 % shape[i]
		idx2 /= shape[i]
	}
}


/**
* Dot product (tensor contraction) between 2 tensors. C=A*B
*
* :param C: output tensor.
* :param A: input tensor 1.
* :param B: input tensor 2.
* :param axesA: Array[naxes] of axes of A being contracted.
* :param axesB: Array[naxes] of axes of B being contracted.
* :param naxes: number of axes being contracted from each input.
* :param normalize: (0,1) whether to L2-normalize samples along the dot product axis before taking the dot product. If set to 1, then the output of the dot product is the cosine proximity between the two samples.
* :param fwork: Array of working space, size(fwork) = size(A) + size(B)
*/
func k2c_dot(C *K2c_tensor, A *K2c_tensor, B *K2c_tensor, axesA []int, axesB []int,
	naxes int, normalize int, fwork []float64) {
	var permA [K2C_MAX_NDIM]int
	var permB [K2C_MAX_NDIM]int
	var prod_axesA = 1
	var prod_axesB = 1
	var free_axesA, free_axesB int
	var freeA [K2C_MAX_NDIM]int
	var freeB [K2C_MAX_NDIM]int
	var count int
	var isin bool
	var newshpA [K2C_MAX_NDIM]int
	var newshpB [K2C_MAX_NDIM]int
	var ndimA = A.Ndim
	var ndimB = B.Ndim
	var reshapeA = fwork // temp working storage
	var reshapeB = fwork[A.Numel:]
	var Asub [K2C_MAX_NDIM]int
	var Bsub [K2C_MAX_NDIM]int
	// find which axes are free (ie, not being summed over)
	count = 0
	for i := 0; i < ndimA; i++ {
		isin = false
		for j := 0; j < naxes; j++ {
			if i == axesA[j] {
				isin = true
				break
			}
		}
		if !isin {
			freeA[count] = i
			count++
		}
	}
	count = 0
	for i := 0; i < ndimB; i++ {
		isin = false
		for j := 0; j < naxes; j++ {
			if i == axesB[j] {
				isin = true
				break
			}
		}
		if !isin {
			freeB[count] = i
			count++
		}
	}
	// number of elements in inner dimension
	for i := 0; i < naxes; i++ {
		prod_axesA *= A.Shape[axesA[i]]
	}
	for i := 0; i < naxes; i++ {
		prod_axesB *= B.Shape[axesB[i]]
	}
	// number of elements in free dimension
	free_axesA = A.Numel / prod_axesA
	free_axesB = B.Numel / prod_axesB
	// find permutation of axes to get into matmul Shape
	for i := 0; i < ndimA-naxes; i++ {
		permA[i] = freeA[i]
	}
	{
		i := ndimA - naxes
		j := 0
		for i < ndimA {
			permA[i] = axesA[j]
			i++
			j++
		}
	}
	for i := 0; i < naxes; i++ {
		permB[i] = axesB[i]
	}
	{
		i := naxes
		j := 0
		for i < ndimB {
			permB[i] = freeB[j]
			i++
			j++
		}
	}

	for i := 0; i < ndimA; i++ {
		newshpA[i] = A.Shape[permA[i]]
	}
	for i := 0; i < ndimB; i++ {
		newshpB[i] = B.Shape[permB[i]]
	}

	// reshape arrays
	for i := 0; i < A.Numel; i++ {
		k2c_idx2sub(i, Asub[:], A.Shape[:], ndimA)
		for j := 0; j < ndimA; j++ {
			Bsub[j] = Asub[permA[j]]
		}
		bidx := k2c_sub2idx(Bsub[:], newshpA[:], ndimA)
		reshapeA[bidx] = A.Array[i]
	}
	for i := 0; i < B.Numel; i++ {
		k2c_idx2sub(i, Bsub[:], B.Shape[:], ndimB)
		for j := 0; j < ndimB; j++ {
			Asub[j] = Bsub[permB[j]]
		}
		bidx := k2c_sub2idx(Asub[:], newshpB[:], ndimB)
		reshapeB[bidx] = B.Array[i]
	}

	if normalize != 0 {
		var sum float64
		var inorm float64
		for i := 0; i < free_axesA; i++ {
			sum = 0
			for j := 0; j < prod_axesA; j++ {
				sum += reshapeA[i*prod_axesA+j] * reshapeA[i*prod_axesA+j]
			}
			inorm = 1.0 / math.Sqrt(sum)
			for j := 0; j < prod_axesA; j++ {
				reshapeA[i*prod_axesA+j] *= inorm
			}
		}
		for i := 0; i < free_axesB; i++ {
			sum = 0
			for j := 0; j < prod_axesB; j++ {
				sum += reshapeB[i+free_axesB*j] * reshapeB[i+free_axesB*j]
			}
			inorm = 1.0 / math.Sqrt(sum)
			for j := 0; j < prod_axesB; j++ {
				reshapeB[i+free_axesB*j] *= inorm
			}
		}
	}

	k2c_matmul(C.Array, reshapeA, reshapeB, free_axesA, free_axesB, prod_axesA)
}


/**
* Adds bias vector b to tensor A.
* assumes b is a rank 1 tensor that is added to the last dimension of A.
*
* :param A: input tensor. Overwritten with outputs.
* :param b: bias tensor.
*/
func k2c_bias_add(A *K2c_tensor, b *K2c_tensor) {
	for i := 0; i < A.Numel; i += b.Numel {
		for j := 0; j < b.Numel; j++ {
			A.Array[i+j] += b.Array[j]
		}
	}
}


/**
* Flips a tensor along specified axis.
* overwrites input with flipped output.
*
* :param A: input tensor. Overwritten with outputs.
* :param axis: axis along which to flip
*/
func k2c_flip(A *K2c_tensor, axis int) {
	var ndim = A.Ndim
	var shape = A.Shape
	var numel = A.Numel
	var sub [K2C_MAX_NDIM]int
	var step = 1
	var k = 0
	var idx = 0
	var temp float64

	var reduced_size = 1
	for i := axis; i < ndim; i++ {
		reduced_size *= shape[i]
	}
	var threshold = reduced_size / 2
	var jump = reduced_size

	for k < numel {
		k2c_idx2sub(k, sub[:], shape[:], ndim)
		sub[axis] = shape[axis] - sub[axis] - 1
		idx = k2c_sub2idx(sub[:], shape[:], ndim)
		temp = A.Array[k]
		A.Array[k] = A.Array[idx]
		A.Array[idx] = temp
		if (k+step)%jump >= threshold {
			k = k + step - threshold + jump
		} else {
			k += step
		}
	}
}
