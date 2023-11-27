package main

import (
	"fmt"
	"math"
	"math/rand"
	"neural-go/nist"
)

type Layer struct {
	Neurons []float64
	Weights [][]float64
	Biases  []float64
}

func (l *Layer) init(size int, nextSize int) {
	l.Neurons = make([]float64, size)
	l.Biases = make([]float64, size)
	l.Weights = make([][]float64, size)
	for i := range l.Weights {
		l.Weights[i] = make([]float64, nextSize)
	}
}

type State struct {
	DeltaWeights [][][]float64
	DeltaBiases  [][]float64
}

type NeuralNetwork struct {
	Layers []Layer
}

func (nn *NeuralNetwork) create(sizes []int) {
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
}

func (nn *NeuralNetwork) feedForward(inputs []float64) []float64 {
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
			nl.Neurons[i] = float64(1) / (float64(1) + math.Exp(-nl.Neurons[i])) // Sigmoid Activation
		}
	}
	return nn.Layers[len(nn.Layers)-1].Neurons
}

func (nn *NeuralNetwork) backPropagation(targets []float64, learningRate float64, moment float64, state *State) {
	if state.DeltaWeights == nil || state.DeltaBiases == nil {
		state.DeltaWeights = make([][][]float64, len(nn.Layers))
		state.DeltaBiases = make([][]float64, len(nn.Layers))
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
			gradients[i] = errors[i] * (cl.Neurons[i] * (float64(1) - cl.Neurons[i])) // Derivative
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
				if state.DeltaWeights[nli] == nil {
					state.DeltaWeights[nli] = make([][]float64, len(cl.Neurons))
					for sdw := range state.DeltaWeights[nli] {
						state.DeltaWeights[nli][sdw] = make([]float64, len(nl.Neurons))
					}
				}

				// Изменение, которое нужно произвести в весе следующего слоя
				deltaWeight := (gradients[i] * nl.Neurons[j] * learningRate) + (moment * state.DeltaWeights[nli][i][j])
				// Обновляем вес
				nl.Weights[j][i] = nl.Weights[j][i] + deltaWeight
				// Сохраняем все дельты в статус, для следующего раза
				state.DeltaWeights[nli][i][j] = deltaWeight
			}
		}

		// Обновляем байесы
		for i := 0; i < len(cl.Neurons); i++ {
			if state.DeltaBiases[cli] == nil {
				state.DeltaBiases[cli] = make([]float64, len(cl.Neurons))
			}
			// Вычисляем дельту для байеса текущего слоя
			deltaBias := (gradients[i] * learningRate) + (moment * state.DeltaBiases[cli][i])
			// Обновляем байес
			cl.Biases[i] += deltaBias
			// Сохраняем все дельты в статус, для следующего раза
			state.DeltaBiases[cli][i] = deltaBias
		}

	}
}

func digits() {
	dataSet, _ := nist.ReadTrainSet("./data")

	imageColors := make([]float64, 28*28)

	var nn = &NeuralNetwork{}
	nn.create([]int{dataSet.W * dataSet.H, 36, 24, 10})

	fmt.Println("START LEARNING")
	var state = &State{}
	right := 0
	nnerror := float64(0)
	for i := 0; i < 200000; i++ {
		learnImageIndex := rand.Int() % 60000
		image := dataSet.Data[learnImageIndex]

		ii := 0
		for _, row := range image.Image {
			for _, pix := range row {
				imageColors[ii] = float64(pix) / 255.0
				ii++
			}
		}

		var a = [10]float64{}
		var target = a[0:10]
		target[image.Digit] = 1.0
		output := nn.feedForward(imageColors)
		maxValue := float64(-1)
		maxIndex := 0
		for i := 0; i < len(output); i++ {
			if output[i] > maxValue {
				maxValue = output[i]
				maxIndex = i
			}
		}
		if maxIndex == image.Digit {
			right++
		}

		for k := 0; k < 10; k++ {
			nnerror += (target[k] - output[k]) * (target[k] - output[k])
		}

		if i%1000 == 0 {
			println("RIGHT: ", right, " ERROR: ", fmt.Sprintf("%f", nnerror))
			right = 0
			nnerror = 0.0
		}

		nn.backPropagation(target, 0.01, 0.1, state)
	}

	fmt.Println("START TESTING")

	dataSet, _ = nist.ReadTestSet("./data")
	right = 0
	for i := 0; i < dataSet.N; i++ {
		image := dataSet.Data[i]
		ii := 0
		for _, row := range image.Image {
			for _, pix := range row {
				imageColors[ii] = float64(pix) / 255.0
				ii++
			}
		}

		output := nn.feedForward(imageColors)
		maxValue := float64(-1)
		maxIndex := 0
		for i := 0; i < len(output); i++ {
			if output[i] > maxValue {
				maxValue = output[i]
				maxIndex = i
			}
		}

		if maxIndex == image.Digit {
			right++
		}

	}

	fmt.Println("Right: ", right)
}

func test() {
	//var nn = NeuralNetwork{}
}

func main() {

	//test()

	digits()
}
