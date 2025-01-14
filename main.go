package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"neural-go/neural"
	"neural-go/nist"
	"os"
)

func main() {
	digits()
}

func digits() {
	nn := load()
	train(&nn)
	test(&nn)
	save(&nn)
	fmt.Println("Exit...")
}

func train(nn *neural.NeuralNetwork) {
	fmt.Println("START LEARNING")

	dataSet, _ := nist.ReadTrainSet("./data")
	imageColors := make([]float64, 28*28)
	right := 0
	nnerror := float64(0)
	for i := 0; i < 100000; i++ {
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
		output := nn.FeedForward(imageColors)
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

		if (i+1)%1000 == 0 {
			println("BATCH: ", i/1000, " RIGHT: ", right, " ERROR: ", fmt.Sprintf("%f", nnerror))
			right = 0
			nnerror = 0.0
		}

		nn.BackPropagation(target, 0.001, 0.5)
	}
}

func test(nn *neural.NeuralNetwork) {
	fmt.Println("START TESTING")
	imageColors := make([]float64, 28*28)
	dataSet, _ := nist.ReadTestSet("./data")
	right := 0
	for i := 0; i < dataSet.N; i++ {
		image := dataSet.Data[i]
		ii := 0
		for _, row := range image.Image {
			for _, pix := range row {
				imageColors[ii] = float64(pix) / 255.0
				ii++
			}
		}

		output := nn.FeedForward(imageColors)
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

func load() neural.NeuralNetwork {
	nn := &neural.NeuralNetwork{}
	jsonDataNN, err := os.ReadFile("data/nn.json")
	if err == nil {
		err1 := json.Unmarshal(jsonDataNN, nn)
		if err1 != nil {
			fmt.Println(err1)
		}
	} else {
		nn.Create([]int{28 * 28, 64, 64, 10})
	}
	return *nn
}

func save(nn *neural.NeuralNetwork) {
	fmt.Print("To save NeuralNetwork press y:")
	key, _ := bufio.NewReader(os.Stdin).ReadByte()
	if key == 121 || key == 89 {
		fmt.Println("Saving...")
		nn.Clean()
		nnJson, _ := json.Marshal(nn)
		err := os.WriteFile("data/nn.json", nnJson, 0644)
		if err != nil {
			fmt.Println(err)
		}
	} else {
		fmt.Println("Do not save!")
	}
}
