import type { NetworkConfig, TrainingConfig } from "./types"

export const defaultNetworkConfig: NetworkConfig = {
  inputSize: 2,
  hiddenLayers: [4, 3],
  outputSize: 1,
  activation: "relu",
  initializer: "glorotUniform",
  networkType: "standard",
  layers: [],
}

export const defaultTrainingConfig: TrainingConfig = {
  learningRate: 0.01,
  epochs: 50,
  batchSize: 32,
  optimizer: "adam",
  lossFunction: "meanSquaredError",
}
