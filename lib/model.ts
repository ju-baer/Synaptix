import * as tf from "@tensorflow/tfjs"
import type { NetworkConfig } from "./types"

export const createModel = (config: NetworkConfig) => {
  // Choose appropriate model creation based on network type
  switch (config.networkType) {
    case "cnn":
      return createCNNModel(config)
    case "rnn":
      return createRNNModel(config)
    default:
      return createStandardModel(config)
  }
}

const createStandardModel = (config: NetworkConfig) => {
  const model = tf.sequential()

  // Input layer
  model.add(
    tf.layers.dense({
      units: config.hiddenLayers[0],
      inputShape: [config.inputSize],
      activation: config.activation,
      kernelInitializer: config.initializer,
    }),
  )

  // Hidden layers
  for (let i = 1; i < config.hiddenLayers.length; i++) {
    model.add(
      tf.layers.dense({
        units: config.hiddenLayers[i],
        activation: config.activation,
        kernelInitializer: config.initializer,
      }),
    )
  }

  // Output layer
  model.add(
    tf.layers.dense({
      units: config.outputSize,
      activation: config.outputSize > 1 ? "softmax" : "sigmoid",
      kernelInitializer: config.initializer,
    }),
  )

  return model
}

const createCNNModel = (config: NetworkConfig) => {
  // If no layers are defined, return null
  if (!config.layers || config.layers.length === 0) {
    return null
  }

  const model = tf.sequential()

  // Add layers based on the configuration
  for (let i = 0; i < config.layers.length; i++) {
    const layer = config.layers[i]
    const layerConfig = { ...layer.config }

    switch (layer.type) {
      case "input":
        // Input layer is implicit in TensorFlow.js, so we don't need to add it
        break

      case "conv2d":
        model.add(
          tf.layers.conv2d({
            inputShape: i === 0 ? layerConfig.shape : undefined,
            filters: layerConfig.filters,
            kernelSize: layerConfig.kernelSize,
            activation: layerConfig.activation,
            padding: "same",
          }),
        )
        break

      case "maxPooling2d":
        model.add(
          tf.layers.maxPooling2d({
            poolSize: [layerConfig.poolSize, layerConfig.poolSize],
            strides: [layerConfig.poolSize, layerConfig.poolSize],
          }),
        )
        break

      case "averagePooling2d":
        model.add(
          tf.layers.averagePooling2d({
            poolSize: [layerConfig.poolSize, layerConfig.poolSize],
            strides: [layerConfig.poolSize, layerConfig.poolSize],
          }),
        )
        break

      case "flatten":
        model.add(tf.layers.flatten())
        break

      case "dense":
        model.add(
          tf.layers.dense({
            units: layerConfig.units,
            activation: layerConfig.activation,
          }),
        )
        break

      case "dropout":
        model.add(tf.layers.dropout({ rate: layerConfig.rate }))
        break
    }
  }

  return model
}

const createRNNModel = (config: NetworkConfig) => {
  // If no layers are defined, return null
  if (!config.layers || config.layers.length === 0) {
    return null
  }

  const model = tf.sequential()

  // Add layers based on the configuration
  for (let i = 0; i < config.layers.length; i++) {
    const layer = config.layers[i]
    const layerConfig = { ...layer.config }

    switch (layer.type) {
      case "input":
        // Input layer is implicit in TensorFlow.js, so we don't need to add it
        break

      case "lstm":
        model.add(
          tf.layers.lstm({
            inputShape: i === 0 ? layerConfig.shape : undefined,
            units: layerConfig.units,
            returnSequences: layerConfig.returnSequences,
          }),
        )
        break

      case "gru":
        model.add(
          tf.layers.gru({
            inputShape: i === 0 ? layerConfig.shape : undefined,
            units: layerConfig.units,
            returnSequences: layerConfig.returnSequences,
          }),
        )
        break

      case "simpleRNN":
        model.add(
          tf.layers.simpleRNN({
            inputShape: i === 0 ? layerConfig.shape : undefined,
            units: layerConfig.units,
            returnSequences: layerConfig.returnSequences,
          }),
        )
        break

      case "dense":
        model.add(
          tf.layers.dense({
            units: layerConfig.units,
            activation: layerConfig.activation,
          }),
        )
        break

      case "dropout":
        model.add(tf.layers.dropout({ rate: layerConfig.rate }))
        break
    }
  }

  return model
}

export const compileModel = (
  model: tf.Sequential,
  config: {
    optimizer: string
    learningRate: number
    lossFunction: string
  },
) => {
  let optimizer

  switch (config.optimizer) {
    case "sgd":
      optimizer = tf.train.sgd(config.learningRate)
      break
    case "adam":
      optimizer = tf.train.adam(config.learningRate)
      break
    case "rmsprop":
      optimizer = tf.train.rmsprop(config.learningRate)
      break
    case "adagrad":
      optimizer = tf.train.adagrad(config.learningRate)
      break
    default:
      optimizer = tf.train.adam(config.learningRate)
  }

  let loss

  switch (config.lossFunction) {
    case "meanSquaredError":
      loss = "meanSquaredError"
      break
    case "categoricalCrossentropy":
      loss = "categoricalCrossentropy"
      break
    case "binaryCrossentropy":
      loss = "binaryCrossentropy"
      break
    default:
      loss = "meanSquaredError"
  }

  model.compile({
    optimizer,
    loss,
    metrics: ["accuracy"],
  })

  return model
}
