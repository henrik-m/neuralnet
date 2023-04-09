using System;

class Program
{
  static void Main(string[] args)
  {
    // // Create a neural network with 2 inputs, 4 hidden nodes, and 1 output
    NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
    // var nn = NeuralNetwork.LoadFromFile("network.txt");

    // // Train the network on the XOR problem
    double[,] inputs = new double[4, 2] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    double[,] targets = new double[4, 1] { { 0 }, { 1 }, { 1 }, { 0 } };
    Train(nn, 30000, inputs, targets);

    // nn.SaveToFile("network.txt");

    // Test the network on the XOR problem
    Test(nn, inputs);
  }

  static void Test(NeuralNetwork nn, double[,] inputs)
  {
    for (int i = 0; i < 4; i++)
    {
      double[] input = new double[2] { inputs[i, 0], inputs[i, 1] };
      double[] output = nn.Forward(input);

      Console.WriteLine("{0} XOR {1} = {2}", inputs[i, 0], inputs[i, 1], output[0]);
    }
  }

  static void Train(NeuralNetwork nn, int numEpochs, double[,] inputs, double[,] targets)
  {
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
      double error = 0;

      for (int i = 0; i < 4; i++)
      {
        double[] input = { inputs[i, 0], inputs[i, 1] };
        double[] target = { targets[i, 0] };

        double[] output = nn.Forward(input);
        nn.Backward(input, target, output, 0.1);

        error += Math.Pow(output[0] - target[0], 2);
      }

      if (epoch % 1000 == 0)
      {
        Console.WriteLine($"Epoch {epoch}: Error = {error:F6}");
      }
    }
  }
}