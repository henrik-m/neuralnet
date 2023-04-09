using System;
using System.IO;

class NeuralNetwork
{
  private int inputSize;
  private int hiddenSize;
  private int outputSize;

  private double[,] weights1;
  private double[] bias1;

  private double[,] weights2;
  private double[] bias2;

  private double[] hidden;

  public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
  {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Initialize weights and biases
    this.weights1 = new double[inputSize, hiddenSize];
    this.bias1 = new double[hiddenSize];
    this.weights2 = new double[hiddenSize, outputSize];
    this.bias2 = new double[outputSize];

    this.hidden = new double[hiddenSize];

    Random random = new Random();

    for (int i = 0; i < inputSize; i++)
    {
      for (int j = 0; j < hiddenSize; j++)
      {
        this.weights1[i, j] = random.NextDouble() * 2 - 1;
      }
    }

    for (int i = 0; i < hiddenSize; i++)
    {
      this.bias1[i] = random.NextDouble() * 2 - 1;
    }

    for (int i = 0; i < hiddenSize; i++)
    {
      for (int j = 0; j < outputSize; j++)
      {
        this.weights2[i, j] = random.NextDouble() * 2 - 1;
      }
    }

    for (int i = 0; i < outputSize; i++)
    {
      this.bias2[i] = random.NextDouble() * 2 - 1;
    }
  }

  public double[] Forward(double[] inputs)
  {
    // Calculate the outputs of the hidden layer
    for (int i = 0; i < hiddenSize; i++)
    {
      double dotProduct = 0;
      for (int j = 0; j < inputSize; j++)
      {
        dotProduct += inputs[j] * weights1[j, i];
      }
      hidden[i] = Sigmoid(dotProduct + bias1[i]);
    }

    // Calculate the outputs of the output layer
    double[] outputs = new double[outputSize];
    for (int i = 0; i < outputSize; i++)
    {
      double dotProduct = 0;
      for (int j = 0; j < hiddenSize; j++)
      {
        dotProduct += hidden[j] * weights2[j, i];
      }
      outputs[i] = Sigmoid(dotProduct + bias2[i]);
    }

    return outputs;
  }

  public void Backward(double[] inputs, double[] targets, double[] outputs, double learningRate)
  {
    // Calculate the error and delta for the output layer
    double[] error = new double[outputSize];
    double[] deltaOutput = new double[outputSize];
    for (int i = 0; i < outputSize; i++)
    {
      error[i] = targets[i] - outputs[i];
      deltaOutput[i] = error[i] * SigmoidPrime(outputs[i]);
    }

    // Calculate the error and delta for the hidden layer
    double[] errorHidden = new double[hiddenSize];
    double[] deltaHidden = new double[hiddenSize];
    for (int i = 0; i < hiddenSize; i++)
    {
      double dotProduct = 0;
      for (int j = 0; j < outputSize; j++)
      {
        dotProduct += deltaOutput[j] * weights2[i, j];
      }
      errorHidden[i] = dotProduct;
      deltaHidden[i] = errorHidden[i] * SigmoidPrime(hidden[i]);
    }

    // Update the weights and biases for the output layer
    for (int i = 0; i < hiddenSize; i++)
    {
      for (int j = 0; j < outputSize; j++)
      {
        weights2[i, j] += learningRate * deltaOutput[j] * hidden[i];
      }
    }

    for (int i = 0; i < outputSize; i++)
    {
      bias2[i] += learningRate * deltaOutput[i];
    }

    // Update the weights and biases for the hidden layer
    for (int i = 0; i < inputSize; i++)
    {
      for (int j = 0; j < hiddenSize; j++)
      {
        weights1[i, j] += learningRate * deltaHidden[j] * inputs[i];
      }
    }

    for (int i = 0; i < hiddenSize; i++)
    {
      bias1[i] += learningRate * deltaHidden[i];
    }
  }

  private double Sigmoid(double x)
  {
    return 1 / (1 + Math.Exp(-x));
  }

  private double SigmoidPrime(double x)
  {
    return Sigmoid(x) * (1 - Sigmoid(x));
  }

  public void SaveToFile(string filePath)
  {
    // Create a new StreamWriter for the file
    using (StreamWriter writer = new StreamWriter(filePath))
    {
      // Write the number of input nodes, hidden nodes, and output nodes to the file
      writer.WriteLine(this.inputSize + " " + this.hiddenSize + " " + this.outputSize);

      // Write the weights from the input layer to the hidden layer to the file
      for (int i = 0; i < this.inputSize; i++)
      {
        for (int j = 0; j < this.hiddenSize; j++)
        {
          writer.Write(this.weights1[i, j] + " ");
        }
        writer.WriteLine();
      }

      // Write the weights from the hidden layer to the output layer to the file
      for (int i = 0; i < this.hiddenSize; i++)
      {
        for (int j = 0; j < this.outputSize; j++)
        {
          writer.Write(this.weights2[i, j] + " ");
        }
        writer.WriteLine();
      }
    }
  }

  public static NeuralNetwork LoadFromFile(string filePath)
  {    

    // Create a new StreamReader for the file
    using (StreamReader reader = new StreamReader(filePath))
    {
      // Read the number of input nodes, hidden nodes, and output nodes from the file
      string[] sizes = reader.ReadLine().Split(' ');

      var nn = new NeuralNetwork(int.Parse(sizes[0]), int.Parse(sizes[1]), int.Parse(sizes[2]));

      // Read the weights from the input layer to the hidden layer from the file
      for (int i = 0; i < nn.inputSize; i++)
      {
        string[] values = reader.ReadLine().Split(' ');
        for (int j = 0; j < nn.hiddenSize; j++)
        {
          nn.weights1[i, j] = double.Parse(values[j]);
        }
      }

      // Read the weights from the hidden layer to the output layer from the file
      for (int i = 0; i < nn.hiddenSize; i++)
      {
        string[] values = reader.ReadLine().Split(' ');
        for (int j = 0; j < nn.outputSize; j++)
        {
          nn.weights2[i, j] = double.Parse(values[j]);
        }
      }

      return nn;
    }
  }
}