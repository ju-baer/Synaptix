import type { DatasetType } from "./types"

export const builtinDatasets: DatasetType[] = [
  {
    id: "xor",
    name: "XOR Problem",
    type: "classification",
    features: [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ],
    labels: [[0], [1], [1], [0]],
    description: "Classic XOR logical operation dataset",
  },
  {
    id: "mnist",
    name: "MNIST Digits",
    type: "classification",
    features: [], // This would be loaded dynamically
    labels: [], // This would be loaded dynamically
    description: "Handwritten digits (subset)",
  },
  {
    id: "iris",
    name: "Iris Flowers",
    type: "classification",
    features: [], // This would be loaded dynamically
    labels: [], // This would be loaded dynamically
    description: "Iris flower species classification",
  },
  {
    id: "sine",
    name: "Sine Wave",
    type: "regression",
    features: [], // This would be generated dynamically
    labels: [], // This would be generated dynamically
    description: "Simple sine wave function approximation",
  },
  {
    id: "imdb_reviews",
    name: "IMDB Reviews",
    type: "sequence_classification",
    features: [], // This would be loaded dynamically
    labels: [], // This would be loaded dynamically
    description: "Sentiment analysis on movie reviews (subset)",
  },
  {
    id: "cifar10",
    name: "CIFAR-10",
    type: "image_classification",
    features: [], // This would be loaded dynamically
    labels: [], // This would be loaded dynamically
    description: "Image classification with 10 categories (subset)",
  },
  {
    id: "time_series",
    name: "Stock Price",
    type: "time_series",
    features: [], // This would be generated dynamically
    labels: [], // This would be generated dynamically
    description: "Stock price prediction dataset",
  },
]
