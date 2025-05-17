export interface NetworkConfig {
  inputSize: number
  hiddenLayers: number[]
  outputSize: number
  activation: string
  initializer: string
  networkType: "standard" | "cnn" | "rnn"
  layers: LayerConfig[]
}

export interface TrainingConfig {
  learningRate: number
  epochs: number
  batchSize: number
  optimizer: string
  lossFunction: string
}

export interface DatasetType {
  id: string
  name: string
  type: string
  features: any[]
  labels: any[]
  description: string
}

export type LayerType =
  | "input"
  | "conv2d"
  | "maxPooling2d"
  | "averagePooling2d"
  | "flatten"
  | "dense"
  | "dropout"
  | "lstm"
  | "gru"
  | "simpleRNN"

export interface LayerConfig {
  type: LayerType
  config: any
}
