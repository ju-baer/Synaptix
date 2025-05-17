export const educationalContent: Record<
  string,
  {
    title: string
    description: string
    sections: {
      title: string
      content: string
    }[]
  }
> = {
  about: {
    title: "About Synaptix",
    description: "An interactive neural network visualization tool",
    sections: [
      {
        title: "Overview",
        content: `
          <p>Synaptix is an educational tool designed to help you understand neural networks through interactive visualization. With Synaptix, you can:</p>
          <ul>
            <li>Build custom neural networks with a drag-and-drop interface</li>
            <li>Visualize the training process in real-time</li>
            <li>Experiment with different network architectures and parameters</li>
            <li>Learn about the inner workings of neural networks</li>
          </ul>
          <p>Whether you're a student, educator, or AI enthusiast, Synaptix provides an intuitive way to explore the concepts behind deep learning.</p>
        `,
      },
      {
        title: "How to Use",
        content: `
          <p>Getting started with Synaptix is easy:</p>
          <ol>
            <li><strong>Build your network</strong>: Use the Network tab to design your neural network architecture.</li>
            <li><strong>Select a dataset</strong>: Choose from built-in datasets or upload your own in the Dataset tab.</li>
            <li><strong>Configure training</strong>: Adjust learning parameters in the Training tab.</li>
            <li><strong>Start training</strong>: Hit the Start Training button and watch your network learn in real-time!</li>
          </ol>
          <p>You can pause training at any time to examine the network state, or adjust parameters on the fly to see how they affect learning.</p>
        `,
      },
      {
        title: "Technology",
        content: `
          <p>Synaptix is built using modern web technologies:</p>
          <ul>
            <li><strong>TensorFlow.js</strong>: For neural network implementation and training</li>
            <li><strong>Three.js</strong>: For 3D visualization of the network</li>
            <li><strong>React</strong>: For the user interface</li>
          </ul>
          <p>All computation happens directly in your browser, so there's no need for server-side processing or installation.</p>
        `,
      },
    ],
  },
  network_architecture: {
    title: "Neural Network Architecture",
    description: "Understanding the structure of neural networks",
    sections: [
      {
        title: "Basics",
        content: `
          <p>A neural network consists of interconnected layers of neurons. The three main types of layers are:</p>
          <ul>
            <li><strong>Input Layer</strong>: Receives the initial data (e.g., image pixels, text features)</li>
            <li><strong>Hidden Layers</strong>: Intermediate layers that perform computations</li>
            <li><strong>Output Layer</strong>: Produces the final result (e.g., classification probabilities)</li>
          </ul>
          <p>Each connection between neurons has a weight, which is adjusted during training to minimize the error.</p>
        `,
      },
      {
        title: "Neurons",
        content: `
          <p>Neurons are the basic units of a neural network. Each neuron:</p>
          <ol>
            <li>Receives inputs from the previous layer</li>
            <li>Applies weights to these inputs</li>
            <li>Sums the weighted inputs</li>
            <li>Applies an activation function</li>
            <li>Passes the result to the next layer</li>
          </ol>
          <p>This simple process, when repeated across many neurons and layers, enables neural networks to learn complex patterns.</p>
        `,
      },
      {
        title: "Design Considerations",
        content: `
          <p>When designing your network architecture, consider:</p>
          <ul>
            <li><strong>Network Depth</strong>: More layers can learn more complex features but may be harder to train</li>
            <li><strong>Layer Width</strong>: More neurons per layer can capture more information but increase computational cost</li>
            <li><strong>Activation Functions</strong>: Different functions (ReLU, Sigmoid, etc.) have different properties</li>
            <li><strong>Weight Initialization</strong>: How weights are initially set can affect training speed and success</li>
          </ul>
          <p>Experiment with different architectures to see what works best for your specific problem!</p>
        `,
      },
    ],
  },
  training_parameters: {
    title: "Training Parameters",
    description: "Key parameters that affect how neural networks learn",
    sections: [
      {
        title: "Overview",
        content: `
          <p>Training a neural network involves adjusting its weights to minimize a loss function. Several parameters control this process:</p>
          <ul>
            <li><strong>Learning Rate</strong>: Controls how much weights change in each update</li>
            <li><strong>Epochs</strong>: Number of complete passes through the training dataset</li>
            <li><strong>Batch Size</strong>: Number of samples processed before weights are updated</li>
            <li><strong>Optimizer</strong>: Algorithm used to update weights (e.g., SGD, Adam)</li>
            <li><strong>Loss Function</strong>: Measures how well the network is performing</li>
          </ul>
          <p>Finding the right combination of these parameters is often key to successful training.</p>
        `,
      },
      {
        title: "Optimization",
        content: `
          <p>Different optimizers have different characteristics:</p>
          <ul>
            <li><strong>SGD (Stochastic Gradient Descent)</strong>: Simple but may converge slowly</li>
            <li><strong>Adam</strong>: Adaptive learning rates, often works well without tuning</li>
            <li><strong>RMSprop</strong>: Good for recurrent networks</li>
            <li><strong>Adagrad</strong>: Adapts learning rates based on parameter frequency</li>
          </ul>
          <p>The best optimizer depends on your specific problem and network architecture.</p>
        `,
      },
      {
        title: "Common Issues",
        content: `
          <p>Watch out for these common training issues:</p>
          <ul>
            <li><strong>Underfitting</strong>: Model is too simple to capture the underlying pattern</li>
            <li><strong>Overfitting</strong>: Model memorizes training data but generalizes poorly</li>
            <li><strong>Vanishing Gradients</strong>: Gradients become too small for effective learning</li>
            <li><strong>Exploding Gradients</strong>: Gradients become too large, causing unstable updates</li>
          </ul>
          <p>Monitoring the loss and accuracy during training can help identify these issues.</p>
        `,
      },
    ],
  },
  visualization: {
    title: "Neural Network Visualization",
    description: "Understanding the 3D visualization of neural networks",
    sections: [
      {
        title: "Elements",
        content: `
          <p>The 3D visualization represents your neural network with these elements:</p>
          <ul>
            <li><strong>Spheres</strong>: Represent neurons in each layer</li>
            <li><strong>Lines</strong>: Represent connections (weights) between neurons</li>
            <li><strong>Colors</strong>: 
              <ul>
                <li>Blue spheres: Input layer neurons</li>
                <li>Gray spheres: Hidden layer neurons</li>
                <li>Green spheres: Output layer neurons</li>
                <li>Orange lines: Active connections during forward/backward pass</li>
              </ul>
            </li>
          </ul>
          <p>You can rotate, zoom, and pan the visualization to examine the network from different angles.</p>
        `,
      },
      {
        title: "Animation",
        content: `
          <p>During training, the visualization animates to show:</p>
          <ul>
            <li><strong>Forward Propagation</strong>: Data flowing from input to output</li>
            <li><strong>Backpropagation</strong>: Error signals flowing backward</li>
            <li><strong>Weight Updates</strong>: Changes in connection strengths</li>
          </ul>
          <p>The animation speed is adjusted to make the process visible, but real neural networks compute much faster!</p>
        `,
      },
      {
        title: "Interpretation",
        content: `
          <p>What to look for in the visualization:</p>
          <ul>
            <li><strong>Activation Patterns</strong>: Which neurons activate for different inputs</li>
            <li><strong>Weight Changes</strong>: How connections strengthen or weaken during training</li>
            <li><strong>Layer Activity</strong>: How information transforms as it passes through layers</li>
          </ul>
          <p>The visualization is simplified for educational purposes. Real neural networks often have thousands or millions of neurons and connections!</p>
        `,
      },
    ],
  },
  datasets: {
    title: "Datasets for Neural Networks",
    description: "Understanding different types of datasets for training",
    sections: [
      {
        title: "Built-in Datasets",
        content: `
          <p>Synaptix includes several built-in datasets for experimentation:</p>
          <ul>
            <li><strong>XOR Problem</strong>: A classic non-linear problem that requires hidden layers</li>
            <li><strong>MNIST Digits</strong>: Handwritten digit recognition (subset of the full dataset)</li>
            <li><strong>Iris Flowers</strong>: Classification of iris flower species based on measurements</li>
            <li><strong>Sine Wave</strong>: Function approximation for a sine wave</li>
          </ul>
          <p>These datasets are designed to demonstrate different aspects of neural network learning.</p>
        `,
      },
      {
        title: "Custom Datasets",
        content: `
          <p>You can upload your own datasets in CSV or JSON format. Requirements:</p>
          <ul>
            <li>Data should be normalized (typically between 0-1 or -1 to 1)</li>
            <li>Features and labels should be clearly separated</li>
            <li>For classification, labels should be one-hot encoded</li>
          </ul>
          <p>Smaller datasets (up to a few thousand samples) work best for interactive visualization.</p>
        `,
      },
      {
        title: "Data Preparation",
        content: `
          <p>Good data preparation is crucial for successful training:</p>
          <ul>
            <li><strong>Normalization</strong>: Scale features to similar ranges</li>
            <li><strong>Feature Selection</strong>: Choose relevant features</li>
            <li><strong>Train/Test Split</strong>: Separate data for training and evaluation</li>
            <li><strong>Shuffling</strong>: Randomize the order of training examples</li>
          </ul>
          <p>Synaptix handles some of these steps automatically, but for custom datasets, preprocessing is recommended.</p>
        `,
      },
    ],
  },
  help: {
    title: "Help & Troubleshooting",
    description: "Common issues and how to resolve them",
    sections: [
      {
        title: "Getting Started",
        content: `
          <p>If you're new to Synaptix, follow these steps:</p>
          <ol>
            <li>Start with a simple network (e.g., 2-4-1 for the XOR problem)</li>
            <li>Choose a built-in dataset like XOR</li>
            <li>Use default training parameters</li>
            <li>Start training and observe the visualization</li>
            <li>Gradually experiment with more complex configurations</li>
          </ol>
          <p>The "About Synaptix" section provides more detailed getting started information.</p>
        `,
      },
      {
        title: "Common Issues",
        content: `
          <p>Troubleshooting common problems:</p>
          <ul>
            <li><strong>Network not learning</strong>: Try increasing the learning rate or adding more hidden neurons</li>
            <li><strong>Training too slow</strong>: Reduce network size or increase batch size</li>
            <li><strong>Browser performance</strong>: Close other tabs or reduce the visualization quality</li>
            <li><strong>Dataset issues</strong>: Ensure your custom dataset is properly formatted</li>
          </ul>
          <p>Remember that neural network training involves randomness, so results may vary between runs.</p>
        `,
      },
      {
        title: "Tips & Tricks",
        content: `
          <p>Advanced tips for using Synaptix effectively:</p>
          <ul>
            <li>Use the pause button to examine network states during training</li>
            <li>Try different activation functions for different problems</li>
            <li>For classification problems, use categorical cross-entropy loss</li>
            <li>For regression problems, use mean squared error loss</li>
            <li>Experiment with learning rate schedules (start high, then decrease)</li>
          </ul>
          <p>The educational overlays provide more in-depth information about specific topics.</p>
        `,
      },
    ],
  },
}

// Add more educational content for other topics
Object.assign(educationalContent, {
  input_layer: {
    title: "Input Layer",
    description: "The first layer of a neural network",
    sections: [
      {
        title: "Purpose",
        content: `
          <p>The input layer is the first layer of a neural network and serves as the interface between your data and the network. Key points:</p>
          <ul>
            <li>Each neuron in the input layer represents one feature of your data</li>
            <li>Input neurons don't perform any computation - they simply pass the input values to the next layer</li>
            <li>The number of input neurons should match the dimensionality of your data</li>
          </ul>
          <p>For example, if you're working with 28×28 pixel images, you would need 784 input neurons (one for each pixel).</p>
        `,
      },
    ],
  },
  hidden_layer: {
    title: "Hidden Layers",
    description: "The intermediate layers in a neural network",
    sections: [
      {
        title: "Purpose",
        content: `
          <p>Hidden layers perform the actual computation in a neural network. They transform the input data through a series of non-linear operations. Key points:</p>
          <ul>
            <li>More hidden layers (deeper networks) can learn more complex patterns</li>
            <li>Each hidden layer typically extracts higher-level features</li>
            <li>The optimal number and size of hidden layers depends on your specific problem</li>
          </ul>
          <p>In practice, start with 1-2 hidden layers and adjust based on performance.</p>
        `,
      },
    ],
  },
  output_layer: {
    title: "Output Layer",
    description: "The final layer of a neural network",
    sections: [
      {
        title: "Purpose",
        content: `
          <p>The output layer produces the final result of the neural network. Its structure depends on your task:</p>
          <ul>
            <li><strong>Classification</strong>: One neuron per class (with softmax activation for multi-class)</li>
            <li><strong>Regression</strong>: Typically one neuron (with linear activation)</li>
            <li><strong>Multi-output regression</strong>: Multiple neurons with linear activation</li>
          </ul>
          <p>The activation function of the output layer should match your problem type.</p>
        `,
      },
    ],
  },
  activation_functions: {
    title: "Activation Functions",
    description: "Non-linear transformations applied to neuron outputs",
    sections: [
      {
        title: "Purpose",
        content: `
          <p>Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Common activation functions:</p>
          <ul>
            <li><strong>ReLU</strong>: f(x) = max(0, x) - Simple and effective, but can suffer from "dying ReLU" problem</li>
            <li><strong>Sigmoid</strong>: f(x) = 1/(1+e^(-x)) - Maps values to range [0,1], but can cause vanishing gradients</li>
            <li><strong>Tanh</strong>: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Maps values to range [-1,1]</li>
            <li><strong>Leaky ReLU</strong>: f(x) = max(αx, x) where α is small - Addresses the dying ReLU problem</li>
          </ul>
          <p>Different activation functions work better for different problems, so experimentation is often necessary.</p>
        `,
      },
    ],
  },
  weight_initializers: {
    title: "Weight Initializers",
    description: "Methods for setting initial weights in neural networks",
    sections: [
      {
        title: "Purpose",
        content: `
          <p>Weight initialization affects how quickly and effectively a neural network learns. Common initializers:</p>
          <ul>
            <li><strong>Glorot (Xavier) Normal/Uniform</strong>: Scales weights based on the number of input and output connections</li>
            <li><strong>He Normal/Uniform</strong>: Similar to Glorot but optimized for ReLU activations</li>
            <li><strong>Random Normal/Uniform</strong>: Simple random initialization, but can lead to vanishing/exploding gradients</li>
          </ul>
          <p>Proper initialization can significantly speed up training and improve final performance.</p>
        `,
      },
    ],
  },
  learning_rate: {
    title: "Learning Rate",
    description: "Controls how quickly weights are updated during training",
    sections: [
      {
        title: "Impact",
        content: `
          <p>The learning rate is one of the most important hyperparameters in neural network training:</p>
          <ul>
            <li><strong>Too high</strong>: Training may diverge or oscillate around the minimum</li>
            <li><strong>Too low</strong>: Training will be slow and may get stuck in local minima</li>
            <li><strong>Just right</strong>: Training converges efficiently to a good solution</li>
          </ul>
          <p>Common values range from 0.1 to 0.0001, with 0.01 being a good starting point for many problems.</p>
        `,
      },
    ],
  },
  epochs: {
    title: "Epochs",
    description: "Number of complete passes through the training dataset",
    sections: [
      {
        title: "Impact",
        content: `
          <p>An epoch represents one complete pass through the entire training dataset:</p>
          <ul>
            <li><strong>Too few</strong>: The model may be undertrained</li>
            <li><strong>Too many</strong>: The model may overfit to the training data</li>
          </ul>
          <p>The optimal number of epochs varies widely depending on the problem, dataset size, and network complexity. Monitoring validation loss helps determine when to stop training.</p>
        `,
      },
    ],
  },
  batch_size: {
    title: "Batch Size",
    description: "Number of samples processed before weights are updated",
    sections: [
      {
        title: "Impact",
        content: `
          <p>Batch size affects both training speed and the quality of weight updates:</p>
          <ul>
            <li><strong>Larger batches</strong>: More stable gradients but require more memory and may generalize poorly</li>
            <li><strong>Smaller batches</strong>: Noisier gradients but can help escape local minima and often generalize better</li>
          </ul>
          <p>Common batch sizes range from 16 to 256, with 32 or 64 being good starting points.</p>
        `,
      },
    ],
  },
  optimizers: {
    title: "Optimizers",
    description: "Algorithms that update network weights during training",
    sections: [
      {
        title: "Types",
        content: `
          <p>Different optimizers have different characteristics:</p>
          <ul>
            <li><strong>SGD (Stochastic Gradient Descent)</strong>: Simple but may converge slowly</li>
            <li><strong>Adam</strong>: Adaptive learning rates, often works well without tuning</li>
            <li><strong>RMSprop</strong>: Good for recurrent networks</li>
            <li><strong>Adagrad</strong>: Adapts learning rates based on parameter frequency</li>
          </ul>
          <p>Adam is often a good default choice for many problems.</p>
        `,
      },
    ],
  },
  loss_functions: {
    title: "Loss Functions",
    description: "Measures how well the network is performing",
    sections: [
      {
        title: "Types",
        content: `
          <p>The loss function should match your problem type:</p>
          <ul>
            <li><strong>Mean Squared Error (MSE)</strong>: For regression problems</li>
            <li><strong>Binary Crossentropy</strong>: For binary classification</li>
            <li><strong>Categorical Crossentropy</strong>: For multi-class classification</li>
          </ul>
          <p>The loss function guides the optimization process by providing a measure of how far the network's predictions are from the true values.</p>
        `,
      },
    ],
  },
  dataset_format: {
    title: "Dataset Format Guidelines",
    description: "How to format custom datasets for Synaptix",
    sections: [
      {
        title: "CSV Format",
        content: `
          <p>For CSV files:</p>
          <ul>
            <li>First row should contain column headers</li>
            <li>Features and labels should be in separate columns</li>
            <li>All values should be numeric</li>
            <li>Missing values should be handled before upload</li>
          </ul>
          <p>Example: For a dataset with 2 features and 1 label:<br>
          <code>feature1,feature2,label<br>
          0.1,0.2,0<br>
          0.3,0.4,1<br>
          ...</code></p>
        `,
      },
      {
        title: "JSON Format",
        content: `
          <p>For JSON files:</p>
          <ul>
            <li>Data should be in an array of objects</li>
            <li>Each object should have "features" and "labels" properties</li>
            <li>"features" and "labels" should be arrays of numbers</li>
          </ul>
          <p>Example:<br>
          <code>[<br>
            {"features": [0.1, 0.2], "labels": [0]},<br>
            {"features": [0.3, 0.4], "labels": [1]},<br>
            ...<br>
          ]</code></p>
        `,
      },
    ],
  },
  dataset_visualization: {
    title: "Dataset Visualization",
    description: "Understanding your data through visualization",
    sections: [
      {
        title: "Features",
        content: `
          <p>Synaptix provides several ways to visualize your dataset:</p>
          <ul>
            <li><strong>Feature Distribution</strong>: Histograms showing the distribution of each feature</li>
            <li><strong>Feature Correlation</strong>: Heatmap showing relationships between features</li>
            <li><strong>Decision Boundaries</strong>: For 2D datasets, shows how the network classifies different regions</li>
            <li><strong>t-SNE Visualization</strong>: For high-dimensional data, shows clusters in a 2D projection</li>
          </ul>
          <p>These visualizations can help you understand your data and identify potential issues before training.</p>
        `,
      },
    ],
  },
  cnn_architecture: {
    title: "Convolutional Neural Network Architecture",
    description: "Understanding the structure and components of CNNs",
    sections: [
      {
        title: "Basics",
        content: `
          <p>Convolutional Neural Networks (CNNs) are specialized architectures designed primarily for processing grid-like data, such as images. Key components include:</p>
          <ul>
            <li><strong>Convolutional Layers</strong>: Extract features by applying filters to the input</li>
            <li><strong>Pooling Layers</strong>: Reduce spatial dimensions and create translation invariance</li>
            <li><strong>Flatten Layer</strong>: Converts multi-dimensional data to a 1D array</li>
            <li><strong>Dense Layers</strong>: Perform classification or regression on extracted features</li>
          </ul>
          <p>CNNs leverage three key ideas: local receptive fields, shared weights, and spatial pooling.</p>
        `,
      },
      {
        title: "Convolutional Layers",
        content: `
          <p>Convolutional layers are the core building blocks of a CNN and work as follows:</p>
          <ul>
            <li><strong>Filters (Kernels)</strong>: Small matrices that slide over the input data</li>
            <li><strong>Feature Maps</strong>: Output produced by applying filters to the input</li>
            <li><strong>Stride</strong>: Number of steps a filter moves in each direction</li>
            <li><strong>Padding</strong>: Adding zeros around the input to control output dimensions</li>
          </ul>
          <p>Each filter learns to detect specific features, like edges, textures, and eventually more complex patterns.</p>
        `,
      },
      {
        title: "Pooling Layers",
        content: `
          <p>Pooling layers perform downsampling operations:</p>
          <ul>
            <li><strong>Max Pooling</strong>: Takes the maximum value within a window</li>
            <li><strong>Average Pooling</strong>: Takes the average value within a window</li>
          </ul>
          <p>Benefits of pooling:</p>
          <ul>
            <li>Reduces computational load</li>
            <li>Controls overfitting</li>
            <li>Provides translation invariance</li>
            <li>Reduces sensitivity to small shifts and distortions</li>
          </ul>
          <p>Typical pooling sizes are 2×2 with a stride of 2, which reduces dimensions by half.</p>
        `,
      },
    ],
  },
  conv2d: {
    title: "Convolutional 2D Layer",
    description: "Extracts features from input data using kernels",
    sections: [
      {
        title: "Overview",
        content: `
          <p>The Convolutional 2D layer applies multiple filters to the input data to extract features. Each filter generates a feature map highlighting specific patterns.</p>
          <p>Key parameters:</p>
          <ul>
            <li><strong>Filters</strong>: Number of feature maps to generate</li>
            <li><strong>Kernel Size</strong>: Size of the convolution window (e.g., 3×3)</li>
            <li><strong>Activation</strong>: Non-linearity applied after convolution</li>
            <li><strong>Padding</strong>: Whether to pad the input to maintain spatial dimensions</li>
            <li><strong>Stride</strong>: Step size when sliding the filter</li>
          </ul>
          <p>Convolutional layers are excellent for capturing spatial patterns in the data.</p>
        `,
      },
    ],
  },
  maxPooling2d: {
    title: "Max Pooling 2D Layer",
    description: "Downsamples feature maps by taking maximum values",
    sections: [
      {
        title: "Overview",
        content: `
          <p>The Max Pooling 2D layer performs downsampling by taking the maximum value from each window of the input feature map.</p>
          <p>Key parameters:</p>
          <ul>
            <li><strong>Pool Size</strong>: Size of the pooling window (e.g., 2×2)</li>
            <li><strong>Stride</strong>: Step size when sliding the window (typically equal to pool size)</li>
          </ul>
          <p>Benefits:</p>
          <ul>
            <li>Reduces spatial dimensions</li>
            <li>Focuses on the most prominent features</li>
            <li>Makes the network more robust to small variations in input</li>
          </ul>
        `,
      },
    ],
  },
  rnn_architecture: {
    title: "Recurrent Neural Network Architecture",
    description: "Understanding the structure and components of RNNs",
    sections: [
      {
        title: "Basics",
        content: `
          <p>Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining a memory of previous inputs. Key characteristics:</p>
          <ul>
            <li><strong>Memory State</strong>: Stores information about past inputs</li>
            <li><strong>Recurrent Connections</strong>: Allow information to flow between time steps</li>
            <li><strong>Sequential Processing</strong>: Process data one step at a time</li>
          </ul>
          <p>RNNs are ideal for tasks like time series forecasting, natural language processing, and speech recognition.</p>
        `,
      },
      {
        title: "RNN Cell Types",
        content: `
          <p>Several types of RNN cells address different challenges:</p>
          <ul>
            <li><strong>Simple RNN</strong>: Basic recurrent structure, but suffers from vanishing/exploding gradients</li>
            <li><strong>LSTM (Long Short-Term Memory)</strong>: Uses gates to control information flow and maintain long-term dependencies</li>
            <li><strong>GRU (Gated Recurrent Unit)</strong>: Simplified version of LSTM with fewer parameters</li>
          </ul>
          <p>LSTMs and GRUs are more commonly used due to their ability to capture long-range dependencies.</p>
        `,
      },
      {
        title: "RNN Configurations",
        content: `
          <p>RNNs can be configured in different ways:</p>
          <ul>
            <li><strong>Many-to-Many</strong>: Input sequence → Output sequence (e.g., machine translation)</li>
            <li><strong>Many-to-One</strong>: Input sequence → Single output (e.g., sentiment analysis)</li>
            <li><strong>One-to-Many</strong>: Single input → Output sequence (e.g., image captioning)</li>
            <li><strong>Bidirectional</strong>: Process sequence in both directions</li>
            <li><strong>Deep RNNs</strong>: Stack multiple recurrent layers</li>
          </ul>
          <p>The choice of configuration depends on the specific task and data characteristics.</p>
        `,
      },
    ],
  },
  lstm: {
    title: "LSTM Layer",
    description: "Long Short-Term Memory recurrent layer",
    sections: [
      {
        title: "Overview",
        content: `
          <p>LSTM (Long Short-Term Memory) is a specialized RNN architecture designed to remember long-term dependencies.</p>
          <p>Key components:</p>
          <ul>
            <li><strong>Cell State</strong>: The memory of the network</li>
            <li><strong>Forget Gate</strong>: Controls what information to discard</li>
            <li><strong>Input Gate</strong>: Controls what new information to store</li>
            <li><strong>Output Gate</strong>: Controls what information to output</li>
          </ul>
          <p>Key parameters:</p>
          <ul>
            <li><strong>Units</strong>: Number of LSTM cells in the layer</li>
            <li><strong>Return Sequences</strong>: Whether to return output for each time step or just the final output</li>
            <li><strong>Recurrent Dropout</strong>: Dropout applied to recurrent connections</li>
          </ul>
          <p>LSTMs excel at tasks requiring memory of events that happened many steps earlier.</p>
        `,
      },
    ],
  },
  gru: {
    title: "GRU Layer",
    description: "Gated Recurrent Unit layer",
    sections: [
      {
        title: "Overview",
        content: `
          <p>GRU (Gated Recurrent Unit) is a simplified variant of LSTM with fewer parameters.</p>
          <p>Key components:</p>
          <ul>
            <li><strong>Update Gate</strong>: Combines forget and input gates from LSTM</li>
            <li><strong>Reset Gate</strong>: Controls how much of the previous state to forget</li>
          </ul>
          <p>Key parameters:</p>
          <ul>
            <li><strong>Units</strong>: Number of GRU cells in the layer</li>
            <li><strong>Return Sequences</strong>: Whether to return output for each time step or just the final output</li>
            <li><strong>Recurrent Dropout</strong>: Dropout applied to recurrent connections</li>
          </ul>
          <p>GRUs often perform similarly to LSTMs but are more computationally efficient and require less data to generalize.</p>
        `,
      },
    ],
  },
  simpleRNN: {
    title: "Simple RNN Layer",
    description: "Basic recurrent neural network layer",
    sections: [
      {
        title: "Overview",
        content: `
          <p>Simple RNN is the most basic form of recurrent neural network.</p>
          <p>Operation:</p>
          <ul>
            <li>Takes current input and previous hidden state</li>
            <li>Applies a linear transformation followed by an activation</li>
            <li>Outputs a new hidden state</li>
          </ul>
          <p>Key parameters:</p>
          <ul>
            <li><strong>Units</strong>: Number of neurons in the layer</li>
            <li><strong>Activation</strong>: Activation function (typically tanh)</li>
            <li><strong>Return Sequences</strong>: Whether to return output for each time step or just the final output</li>
          </ul>
          <p>Limitations:</p>
          <ul>
            <li>Struggles with long-term dependencies due to vanishing/exploding gradients</li>
            <li>Generally outperformed by LSTM and GRU for most tasks</li>
          </ul>
          <p>Simple RNNs are primarily used for educational purposes or very simple sequence tasks with short-term dependencies.</p>
        `,
      },
    ],
  },
})
