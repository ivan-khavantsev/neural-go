package neural

import (
	"math"
	"math/rand"
)

func Sigmoid(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (float64(1) - x)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhDerivative(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

type NeuralNetwork struct {
	Layers []Layer
	State  *State
}

type Layer struct {
	Neurons []float64
	Weights [][]float64
	Biases  []float64
}

type State struct {
	DeltaWeights [][][]float64
	DeltaBiases  [][]float64
}

func (l *Layer) init(size int, nextSize int) {
	l.Neurons = make([]float64, size)
	l.Biases = make([]float64, size)
	l.Weights = make([][]float64, size)
	for i := range l.Weights {
		l.Weights[i] = make([]float64, nextSize)
	}
}

func (nn *NeuralNetwork) Create(sizes []int) {
	nn.Layers = make([]Layer, len(sizes))
	for i := 0; i < len(sizes); i++ {
		size := sizes[i]
		nextSize := 0
		if i < len(sizes)-1 {
			nextSize = sizes[i+1]
		}
		var layer Layer
		layer.init(size, nextSize)
		nn.Layers[i] = layer

		for j := 0; j < sizes[i]; j++ {
			nn.Layers[i].Biases[j] = rand.Float64()*2.0 - 1.0
			for k := 0; k < nextSize; k++ {
				nn.Layers[i].Weights[j][k] = rand.Float64()*2.0 - 1.0
			}
		}
	}
	nn.State = &State{}
}

func (nn *NeuralNetwork) FeedForward(inputs []float64) []float64 {
	for i := range inputs {
		nn.Layers[0].Neurons[i] = inputs[i]
	}
	for l := 1; l < len(nn.Layers); l++ {
		cl := &nn.Layers[l-1]
		nl := &nn.Layers[l]
		for i := 0; i < len(nl.Neurons); i++ {
			nl.Neurons[i] = 0
			for j := 0; j < len(cl.Neurons); j++ {
				nl.Neurons[i] += cl.Neurons[j] * cl.Weights[j][i]
			}
			nl.Neurons[i] += nl.Biases[i]
			nl.Neurons[i] = Sigmoid(nl.Neurons[i])
		}
	}
	return nn.Layers[len(nn.Layers)-1].Neurons
}

// Очистить все значения нейронов, например для сохранения сети
func (nn *NeuralNetwork) Clean() {
	for i := range nn.Layers {
		for j := range nn.Layers[i].Neurons {
			nn.Layers[i].Neurons[j] = 0
		}
	}
}

func (nn *NeuralNetwork) BackPropagation(targets []float64, learningRate float64, moment float64) {
	if nn.State.DeltaWeights == nil || nn.State.DeltaBiases == nil {
		nn.State.DeltaWeights = make([][][]float64, len(nn.Layers))
		nn.State.DeltaBiases = make([][]float64, len(nn.Layers))
	}
	ol := &nn.Layers[len(nn.Layers)-1]

	errors := make([]float64, len(ol.Neurons))
	for i := range errors {
		errors[i] = targets[i] - ol.Neurons[i]
	}

	for nli := len(nn.Layers) - 2; nli >= 0; nli-- {
		nl := &nn.Layers[nli]
		cli := nli + 1
		cl := &nn.Layers[cli]

		gradients := make([]float64, len(cl.Neurons))
		for i := 0; i < len(cl.Neurons); i++ {
			gradients[i] = errors[i] * SigmoidDerivative(cl.Neurons[i])
		}

		errorsNext := make([]float64, len(nl.Neurons))
		for i := 0; i < len(nl.Neurons); i++ {
			for j := 0; j < len(cl.Neurons); j++ {
				errorsNext[i] += nl.Weights[i][j] * errors[j]
			}
		}
		errors = make([]float64, len(nl.Neurons))
		for i := range errorsNext {
			errors[i] = errorsNext[i]
		}

		// Вычисляем и устанавливаем новые веса для следующего слоя
		for i := 0; i < len(cl.Neurons); i++ {
			for j := 0; j < len(nl.Neurons); j++ {
				if nn.State.DeltaWeights[nli] == nil {
					nn.State.DeltaWeights[nli] = make([][]float64, len(cl.Neurons))
					for sdw := range nn.State.DeltaWeights[nli] {
						nn.State.DeltaWeights[nli][sdw] = make([]float64, len(nl.Neurons))
					}
				}

				// Изменение, которое нужно произвести в весе следующего слоя
				deltaWeight := (gradients[i] * nl.Neurons[j] * learningRate) + (moment * nn.State.DeltaWeights[nli][i][j])
				// Обновляем вес
				nl.Weights[j][i] = nl.Weights[j][i] + deltaWeight
				// Сохраняем все дельты в статус, для следующего раза
				nn.State.DeltaWeights[nli][i][j] = deltaWeight
			}
		}

		// Обновляем байесы
		for i := 0; i < len(cl.Neurons); i++ {
			if nn.State.DeltaBiases[cli] == nil {
				nn.State.DeltaBiases[cli] = make([]float64, len(cl.Neurons))
			}
			// Вычисляем дельту для байеса текущего слоя
			deltaBias := (gradients[i] * learningRate) + (moment * nn.State.DeltaBiases[cli][i])
			// Обновляем байес
			cl.Biases[i] += deltaBias
			// Сохраняем все дельты в статус, для следующего раза
			nn.State.DeltaBiases[cli][i] = deltaBias
		}

	}
}
