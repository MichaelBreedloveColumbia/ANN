using System;
using System.Collections.Generic;
using System.Linq;

namespace simple_neural_network
{
    class NeuralNetWork
    {
        private Random _radomObj;

        public NeuralNetWork(int synapseMatrixColumns, int synapseMatrixLines)
        {
            SynapseMatrixColumns = synapseMatrixColumns;
            SynapseMatrixLines = synapseMatrixLines;

            _Init();
        }

        public int SynapseMatrixColumns { get; }
        public int SynapseMatrixLines { get; }
        public double[,] SynapsesMatrix { get; private set; }

        /// <summary>
        /// Initialize the ramdom object and the matrix of ramdon weights
        /// </summary>
        private void _Init()
        {
            // make sure that for every instance of the neural network we are geting the same radom values
            _radomObj = new Random(1);
            _GenerateSynapsesMatrix();
        }

        /// <summary>
        /// Generate our matrix with the weight of the synapses
        /// </summary>
        private void _GenerateSynapsesMatrix()
        {
            SynapsesMatrix = new double[SynapseMatrixLines, SynapseMatrixColumns];

            for (var i = 0; i < SynapseMatrixLines; i++)
            {
                for (var j = 0; j < SynapseMatrixColumns; j++)
                {
                    SynapsesMatrix[i, j] = (2 * _radomObj.NextDouble()) - 1;
                }
            }
        }

        /// <summary>
        /// Calculate the sigmoid of a value
        /// </summary>
        /// <returns></returns>
        private double[,] _CalculateSigmoid(double[,] matrix)
        {

            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    var value = matrix[i, j];
                    matrix[i, j] = 1 / (1 + Math.Exp(value * -1));
                }
            }
            return matrix;
        }

        /// <summary>
        /// Calculate the sigmoid derivative of a value
        /// </summary>
        /// <returns></returns>
        private double[,] _CalculateSigmoidDerivative(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    var value = matrix[i, j];
                    matrix[i, j] = value * (1 - value);
                }
            }
            return matrix;
        }

        /// <summary>
        /// Will return the outputs give the set of the inputs
        /// </summary>
        public double[,] Think(double[,] inputMatrix)
        {
            var productOfTheInputsAndWeights = MatrixDotProduct(inputMatrix, SynapsesMatrix);

            return _CalculateSigmoid(productOfTheInputsAndWeights);

        }

        /// <summary>
        /// Train the neural network to achieve the output matrix values
        /// </summary>
        public void Train(double[,] trainInputMatrix, double[,] trainOutputMatrix, int interactions)
        {
            // we run all the interactions
            for (var i = 0; i < interactions; i++)
            {
                // calculate the output
                var output = Think(trainInputMatrix);

                // calculate the error
                var error = MatrixSubstract(trainOutputMatrix, output);
                var curSigmoidDerivative = _CalculateSigmoidDerivative(output);
                var error_SigmoidDerivative = MatrixProduct(error, curSigmoidDerivative);

                // calculate the adjustment :) 
                var adjustment = MatrixDotProduct(MatrixTranspose(trainInputMatrix), error_SigmoidDerivative);

                SynapsesMatrix = MatrixSum(SynapsesMatrix, adjustment);
            }
        }

        /// <summary>
        /// Transpose a matrix
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixTranspose(double[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            double[,] result = new double[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Sum one matrix with another
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixSum(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] + matrixb[i, u];
                }
            }

            return result;
        }

        /// <summary>
        /// Subtract one matrix from another
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixSubstract(double[,] matrixa, double[,] matrixb) //Something in here crashes on the second run through. This shouldn't happen, but it does anyway and I do not know why.
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] - matrixb[i, u];
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplication of a matrix
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixProduct(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] * matrixb[i, u];
                }
            }

            return result;
        }

        /// <summary>
        /// Dot Multiplication of a matrix
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixDotProduct(double[,] matrixa, double[,] matrixb)
        {

            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var rowsB = matrixb.GetLength(0);
            var colsB = matrixb.GetLength(1);

            if (colsA != rowsB)
                throw new Exception("Matrices dimensions don't fit.");

            var result = new double[rowsA, colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    for (int k = 0; k < rowsB; k++)
                        result[i, j] += matrixa[i, k] * matrixb[k, j];
                }
            }
            return result;
        }

    }

    class HypotheticalCharacter
    {
        public float defense = 0f;

        public HypotheticalCharacter(float helmet, float chestplate, float leggings, float boots)
        {
            Console.WriteLine($"The player's defense is {defense}.");
            Console.WriteLine($"The player equipped a helmet which adjusted their defense by {helmet}.");
            Console.WriteLine($"The player equipped a chestplate which adjusted their defense by {chestplate}.");
            Console.WriteLine($"The player equipped a pair of leggings which adjusted their defense by {leggings}.");
            Console.WriteLine($"The player equipped a pair of boots which adjusted their defense by {boots}.");
            defense += helmet + chestplate + leggings + boots;
            Console.WriteLine($"The player's new defense is {defense}.");
        }
    }

    class Program
    {

        static void PrintMatrix(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine);
            }
        }

        static List<float> RetrieveWeights(double[,] matrix)
        {
            List<float> weights = new List<float>();

            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    weights.Add((float)matrix[i, j]);
                }
            }

            return weights;
        }

        class NetworkRunner
        {
            public NetworkRunner()
            {
                RunNetwork(new double[,] { { 0, 0, 1, 1 }, { 1, 1, 1, 1 }, { 1, 0, 1, 0 }, { 0, 1, 1, 0 } }, new double[,] { { 0, 1, 1, 0 } }, 0f,
                    new double[,] { { 0, 0, 1, 1 }, { 1, 1, 1, 1 }, { 1, 0, 1, 0 }, { 0, 1, 1, 0 } });
            }

            void RunNetwork(double[,] inputs, double[,] outputs, float oldDefense, double[,] oldInputs)
            {
                Console.Clear();

                var curNeuralNetwork = new NeuralNetWork(1, 4);

                Console.WriteLine("Armor stat modifiers before training:");
                PrintMatrix(curNeuralNetwork.SynapsesMatrix);

                var trainingInputs = inputs;
                var trainingOutputs = NeuralNetWork.MatrixTranspose(outputs);

                curNeuralNetwork.Train(trainingInputs, trainingOutputs, 10000);

                Console.WriteLine("\nArmor stat modifiers after training:");
                PrintMatrix(curNeuralNetwork.SynapsesMatrix);

                Console.WriteLine("The boss is readying a very powerful attack which will deal 30 damage.");

                List<float> weights = RetrieveWeights(curNeuralNetwork.SynapsesMatrix);
                HypotheticalCharacter character = new HypotheticalCharacter(weights[0], weights[1], weights[2], weights[3]);
                if (character.defense < 30f)
                {
                    Console.WriteLine("The player did not survive the attack.");
                }
                else
                {
                    Console.WriteLine("The player survived the attack.");
                }

                Console.WriteLine("Run again? [Y/N]");
                if (Console.ReadLine().ToUpper() == "Y")
                {
                    trainingInputs = AdjustMatrix(trainingInputs, oldInputs, character.defense > oldDefense);
                    //trainingOutputs = AdjustMatrix(trainingOutputs, oldOutputs, character.defense > oldDefense);

                    RunNetwork(trainingInputs, trainingOutputs, character.defense, inputs);
                }
                else
                {
                    Environment.Exit(0);
                }
            }

            double[,] AdjustMatrix(double[,] matrix, double[,] compareMatrix, bool success)
            {
                int rowLength = matrix.GetLength(0);
                int colLength = matrix.GetLength(1);

                int rowLengthCompare = compareMatrix.GetLength(0);
                int colLengthCompare = compareMatrix.GetLength(1);

                if (rowLength != rowLengthCompare || colLength != colLengthCompare)
                {
                    Console.WriteLine("Failed to adjust matrix due to inequality in scale.");
                    Environment.Exit(0);
                }

                Random rand = new Random();
                for (int i = 0; i < rowLength; i++)
                {
                    for (int j = 0; j < colLength; j++)
                    {
                        if (matrix[i, j] > compareMatrix[i, j])
                        {
                            matrix[i, j] += (float)rand.NextDouble() * (success ? 1 : -1);
                        }
                        else if (matrix[i, j] < compareMatrix[i, j])
                        {
                            matrix[i, j] -= (float)rand.NextDouble() * (success ? 1 : -1);
                        }
                        else
                        {
                            matrix[i, j] += (float)rand.Next(-1, 1);
                        }
                    }
                }

                return matrix;
            }
        }

        static void Main(string[] args)
        {
            NetworkRunner runner = new NetworkRunner();
        }
    }
}
