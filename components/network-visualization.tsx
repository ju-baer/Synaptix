"use client"

import { useRef, useEffect, useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Text } from "@react-three/drei"
import type { NetworkConfig } from "@/lib/types"
import type * as THREE from "three"

interface NetworkVisualizationProps {
  networkConfig: NetworkConfig
  isTraining: boolean
  isPaused: boolean
  currentEpoch: number
}

const NetworkVisualization = ({ networkConfig, isTraining, isPaused, currentEpoch }: NetworkVisualizationProps) => {
  const [visualizationType, setVisualizationType] = useState(networkConfig.networkType)

  // Update visualization type when network type changes
  useEffect(() => {
    setVisualizationType(networkConfig.networkType)
  }, [networkConfig.networkType])

  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [0, 0, 15], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />

        {visualizationType === "standard" && (
          <StandardNeuralNetwork
            networkConfig={networkConfig}
            isTraining={isTraining}
            isPaused={isPaused}
            currentEpoch={currentEpoch}
          />
        )}

        {visualizationType === "cnn" && (
          <ConvolutionalNeuralNetwork
            networkConfig={networkConfig}
            isTraining={isTraining}
            isPaused={isPaused}
            currentEpoch={currentEpoch}
          />
        )}

        {visualizationType === "rnn" && (
          <RecurrentNeuralNetwork
            networkConfig={networkConfig}
            isTraining={isTraining}
            isPaused={isPaused}
            currentEpoch={currentEpoch}
          />
        )}

        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
    </div>
  )
}

interface NeuralNetworkProps {
  networkConfig: NetworkConfig
  isTraining: boolean
  isPaused: boolean
  currentEpoch: number
}

const StandardNeuralNetwork = ({ networkConfig, isTraining, isPaused, currentEpoch }: NeuralNetworkProps) => {
  const groupRef = useRef<THREE.Group>(null)

  // Calculate positions for all layers
  const layers = [networkConfig.inputSize, ...networkConfig.hiddenLayers, networkConfig.outputSize]

  const maxNeurons = Math.max(...layers)
  const layerSpacing = 4
  const neuronSpacing = 1.5

  useEffect(() => {
    if (isTraining && !isPaused && groupRef.current) {
      // Animation logic would go here
    }
  }, [isTraining, isPaused, currentEpoch])

  return (
    <group ref={groupRef}>
      {/* Render layers */}
      {layers.map((neuronCount, layerIndex) => {
        const xPos = layerIndex * layerSpacing - ((layers.length - 1) * layerSpacing) / 2

        return (
          <group key={`layer-${layerIndex}`} position={[xPos, 0, 0]}>
            {/* Layer label */}
            <Text
              position={[0, (maxNeurons * neuronSpacing) / 2 + 1.5, 0]}
              fontSize={0.5}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {layerIndex === 0
                ? "Input Layer"
                : layerIndex === layers.length - 1
                  ? "Output Layer"
                  : `Hidden Layer ${layerIndex}`}
            </Text>

            {/* Neurons */}
            {Array.from({ length: neuronCount }).map((_, neuronIndex) => {
              const yPos = (neuronIndex - (neuronCount - 1) / 2) * neuronSpacing

              return (
                <group key={`neuron-${layerIndex}-${neuronIndex}`} position={[0, yPos, 0]}>
                  <mesh>
                    <sphereGeometry args={[0.4, 32, 32]} />
                    <meshStandardMaterial
                      color={
                        layerIndex === 0
                          ? "#1a202c" // Input layer - dark gray/black
                          : layerIndex === layers.length - 1
                            ? "#2d3748" // Output layer - darker gray
                            : "#4a5568" // Hidden layers - medium gray
                      }
                    />
                  </mesh>

                  {/* Connections to next layer */}
                  {layerIndex < layers.length - 1 && (
                    <>
                      {Array.from({ length: layers[layerIndex + 1] }).map((_, nextNeuronIndex) => {
                        const nextYPos = (nextNeuronIndex - (layers[layerIndex + 1] - 1) / 2) * neuronSpacing

                        return (
                          <Connection
                            key={`connection-${layerIndex}-${neuronIndex}-${nextNeuronIndex}`}
                            startPosition={[0, yPos, 0]}
                            endPosition={[layerSpacing, nextYPos, 0]}
                            isActive={isTraining && !isPaused}
                          />
                        )
                      })}
                    </>
                  )}
                </group>
              )
            })}
          </group>
        )
      })}

      {/* Training status */}
      {isTraining && (
        <Text
          position={[0, (-maxNeurons * neuronSpacing) / 2 - 2, 0]}
          fontSize={0.6}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {isPaused ? "Training Paused" : `Training - Epoch ${currentEpoch}`}
        </Text>
      )}
    </group>
  )
}

const ConvolutionalNeuralNetwork = ({ networkConfig, isTraining, isPaused, currentEpoch }: NeuralNetworkProps) => {
  const groupRef = useRef<THREE.Group>(null)

  // Get input shape from the input layer if available
  const inputShape = networkConfig.layers[0]?.config?.shape || [28, 28, 1]

  // Calculate positions based on layers
  const layerSpacing = 5
  const layerConfigs = processConvolutionalLayers(networkConfig.layers)

  useEffect(() => {
    if (isTraining && !isPaused && groupRef.current) {
      // Animation logic would go here
    }
  }, [isTraining, isPaused, currentEpoch])

  return (
    <group ref={groupRef}>
      {/* Render CNN layers */}
      {layerConfigs.map((layer, index) => {
        const xPos = index * layerSpacing - ((layerConfigs.length - 1) * layerSpacing) / 2

        return (
          <group key={`cnn-layer-${index}`} position={[xPos, 0, 0]}>
            {/* Layer label */}
            <Text position={[0, 3.5, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
              {layer.name}
            </Text>

            {/* Layer visualization */}
            {layer.type === "input" && <InputLayer dimensions={layer.dimensions} color="#1a202c" />}

            {layer.type === "conv2d" && (
              <ConvolutionalLayer dimensions={layer.dimensions} filters={layer.filters} color="#2d3748" />
            )}

            {(layer.type === "maxPooling2d" || layer.type === "averagePooling2d") && (
              <PoolingLayer dimensions={layer.dimensions} color="#4a5568" />
            )}

            {layer.type === "flatten" && <FlattenLayer dimensions={layer.dimensions} color="#718096" />}

            {layer.type === "dense" && (
              <DenseLayer neurons={layer.neurons} color={index === layerConfigs.length - 1 ? "#2d3748" : "#4a5568"} />
            )}

            {/* Connections to next layer */}
            {index < layerConfigs.length - 1 && (
              <LayerConnection
                startLayer={layer}
                endLayer={layerConfigs[index + 1]}
                distance={layerSpacing}
                isActive={isTraining && !isPaused}
              />
            )}
          </group>
        )
      })}

      {/* Training status */}
      {isTraining && (
        <Text position={[0, -4, 0]} fontSize={0.6} color="white" anchorX="center" anchorY="middle">
          {isPaused ? "Training Paused" : `Training - Epoch ${currentEpoch}`}
        </Text>
      )}
    </group>
  )
}

const RecurrentNeuralNetwork = ({ networkConfig, isTraining, isPaused, currentEpoch }: NeuralNetworkProps) => {
  const groupRef = useRef<THREE.Group>(null)

  // Process RNN layers for visualization
  const layerConfigs = processRecurrentLayers(networkConfig.layers)
  const layerSpacing = 5
  const timeSteps = 5 // For visualization purposes, show 5 time steps

  useEffect(() => {
    if (isTraining && !isPaused && groupRef.current) {
      // Animation logic would go here
    }
  }, [isTraining, isPaused, currentEpoch])

  return (
    <group ref={groupRef}>
      {/* Render RNN layers */}
      {layerConfigs.map((layer, layerIndex) => {
        const xPos = layerIndex * layerSpacing - ((layerConfigs.length - 1) * layerSpacing) / 2

        return (
          <group key={`rnn-layer-${layerIndex}`} position={[xPos, 0, 0]}>
            {/* Layer label */}
            <Text position={[0, 4, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
              {layer.name}
            </Text>

            {/* Layer visualization */}
            {layer.type === "input" && (
              <RNNInputLayer timeSteps={timeSteps} dimensions={layer.dimensions} color="#1a202c" />
            )}

            {(layer.type === "lstm" || layer.type === "gru" || layer.type === "simpleRNN") && (
              <RecurrentLayer
                timeSteps={timeSteps}
                units={layer.units}
                returnSequences={layer.returnSequences}
                color="#2d3748"
              />
            )}

            {layer.type === "dense" && (
              <DenseLayer
                neurons={layer.neurons}
                color={layerIndex === layerConfigs.length - 1 ? "#2d3748" : "#4a5568"}
              />
            )}

            {/* Connections to next layer */}
            {layerIndex < layerConfigs.length - 1 && (
              <RNNLayerConnection
                startLayer={layer}
                endLayer={layerConfigs[layerIndex + 1]}
                distance={layerSpacing}
                isActive={isTraining && !isPaused}
              />
            )}
          </group>
        )
      })}

      {/* Training status */}
      {isTraining && (
        <Text position={[0, -4, 0]} fontSize={0.6} color="white" anchorX="center" anchorY="middle">
          {isPaused ? "Training Paused" : `Training - Epoch ${currentEpoch}`}
        </Text>
      )}
    </group>
  )
}

// Process CNN layers for visualization
function processConvolutionalLayers(layers: any[]) {
  if (!layers || layers.length === 0) {
    // Default CNN architecture if no layers provided
    return [
      { type: "input", name: "Input", dimensions: [28, 28, 1] },
      { type: "conv2d", name: "Conv2D (16)", dimensions: [26, 26, 16], filters: 16 },
      { type: "maxPooling2d", name: "MaxPooling2D", dimensions: [13, 13, 16] },
      { type: "conv2d", name: "Conv2D (32)", dimensions: [11, 11, 32], filters: 32 },
      { type: "maxPooling2d", name: "MaxPooling2D", dimensions: [5, 5, 32] },
      { type: "flatten", name: "Flatten", dimensions: [800] },
      { type: "dense", name: "Dense (64)", neurons: 64 },
      { type: "dense", name: "Output (10)", neurons: 10 },
    ]
  }

  // Process layers based on their configurations
  const result = []
  let currentDimensions: number[] = []

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i]
    const layerConfig = { ...layer.config }

    switch (layer.type) {
      case "input":
        currentDimensions = layerConfig.shape
        result.push({
          type: "input",
          name: "Input",
          dimensions: currentDimensions,
        })
        break

      case "conv2d":
        // Calculate output dimensions for conv layer
        const filters = layerConfig.filters
        const kernelSize = layerConfig.kernelSize
        const newWidth = currentDimensions[0] - kernelSize + 1
        const newHeight = currentDimensions[1] - kernelSize + 1
        currentDimensions = [newWidth, newHeight, filters]

        result.push({
          type: "conv2d",
          name: `Conv2D (${filters})`,
          dimensions: currentDimensions,
          filters,
        })
        break

      case "maxPooling2d":
      case "averagePooling2d":
        // Calculate output dimensions for pooling layer
        const poolSize = layerConfig.poolSize
        const pooledWidth = Math.floor(currentDimensions[0] / poolSize)
        const pooledHeight = Math.floor(currentDimensions[1] / poolSize)
        currentDimensions = [pooledWidth, pooledHeight, currentDimensions[2]]

        result.push({
          type: layer.type,
          name: layer.type === "maxPooling2d" ? "MaxPooling2D" : "AvgPooling2D",
          dimensions: currentDimensions,
        })
        break

      case "flatten":
        // Calculate flattened size
        const flattenedSize = currentDimensions.reduce((a, b) => a * b, 1)
        currentDimensions = [flattenedSize]

        result.push({
          type: "flatten",
          name: "Flatten",
          dimensions: currentDimensions,
        })
        break

      case "dense":
        const units = layerConfig.units
        currentDimensions = [units]

        result.push({
          type: "dense",
          name: i === layers.length - 1 ? `Output (${units})` : `Dense (${units})`,
          dimensions: currentDimensions,
          neurons: units,
        })
        break

      case "dropout":
        // Dropout doesn't change dimensions, so we skip it in visualization
        break
    }
  }

  return result
}

// Process RNN layers for visualization
function processRecurrentLayers(layers: any[]) {
  if (!layers || layers.length === 0) {
    // Default RNN architecture if no layers provided
    return [
      { type: "input", name: "Input", dimensions: [10, 1] },
      { type: "lstm", name: "LSTM (32)", units: 32, returnSequences: false },
      { type: "dense", name: "Dense (16)", neurons: 16 },
      { type: "dense", name: "Output (1)", neurons: 1 },
    ]
  }

  // Process layers based on their configurations
  const result = []
  let hasSequence = true

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i]
    const layerConfig = { ...layer.config }

    switch (layer.type) {
      case "input":
        const shape = layerConfig.shape
        result.push({
          type: "input",
          name: "Input",
          dimensions: shape,
        })
        break

      case "lstm":
      case "gru":
      case "simpleRNN":
        const units = layerConfig.units
        const returnSequences = layerConfig.returnSequences
        hasSequence = returnSequences

        result.push({
          type: layer.type,
          name: `${layer.type.toUpperCase()} (${units})`,
          units,
          returnSequences,
        })
        break

      case "dense":
        const denseUnits = layerConfig.units

        result.push({
          type: "dense",
          name: i === layers.length - 1 ? `Output (${denseUnits})` : `Dense (${denseUnits})`,
          neurons: denseUnits,
        })
        break

      case "dropout":
        // Dropout doesn't change dimensions, so we skip it in visualization
        break
    }
  }

  return result
}

// Component for connections between neurons
interface ConnectionProps {
  startPosition: [number, number, number]
  endPosition: [number, number, number]
  isActive: boolean
}

const Connection = ({ startPosition, endPosition, isActive }: ConnectionProps) => {
  const [x1, y1, z1] = startPosition
  const [x2, y2, z2] = endPosition

  // Calculate the midpoint and direction
  const direction = [x2 - x1, y2 - y1, z2 - z1]
  const length = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)

  // Calculate rotation to point cylinder in the right direction
  const rotationY = Math.atan2(direction[0], direction[2])
  const rotationZ = Math.atan2(direction[1], Math.sqrt(direction[0] ** 2 + direction[2] ** 2))

  return (
    <mesh position={[(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]} rotation={[0, rotationY, rotationZ]}>
      <cylinderGeometry args={[0.03, 0.03, length, 8]} />
      <meshStandardMaterial color={isActive ? "#718096" : "#2d3748"} opacity={0.7} transparent={true} />
    </mesh>
  )
}

// Input layer visualization for CNN
const InputLayer = ({ dimensions, color }: { dimensions: number[]; color: string }) => {
  const [width, height, channels] = dimensions
  const scale = 0.1 // Scale to fit in view
  const size = Math.max(width, height) * scale

  return (
    <group>
      <mesh>
        <boxGeometry args={[(size * width) / Math.max(width, height), size, 0.2]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {`${width}x${height}x${channels}`}
      </Text>
    </group>
  )
}

// Convolutional layer visualization
const ConvolutionalLayer = ({
  dimensions,
  filters,
  color,
}: { dimensions: number[]; filters: number; color: string }) => {
  const [width, height, channels] = dimensions
  const scale = 0.1
  const size = Math.max(width, height) * scale

  // Create multiple planes to represent feature maps
  return (
    <group>
      {Array.from({ length: Math.min(5, filters) }).map((_, i) => {
        const offset = (i - Math.min(5, filters) / 2 + 0.5) * 0.3
        return (
          <mesh key={i} position={[0, offset, 0]}>
            <boxGeometry
              args={[(size * width) / Math.max(width, height), (size * height) / Math.max(width, height), 0.05]}
            />
            <meshStandardMaterial color={color} opacity={0.8} transparent={true} />
          </mesh>
        )
      })}

      {filters > 5 && (
        <Text position={[0, -1.5, 0]} fontSize={0.3} color="white" anchorX="center" anchorY="middle">
          {`...${filters - 5} more`}
        </Text>
      )}

      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {`${width}x${height}x${channels}`}
      </Text>
    </group>
  )
}

// Pooling layer visualization
const PoolingLayer = ({ dimensions, color }: { dimensions: number[]; color: string }) => {
  const [width, height, channels] = dimensions
  const scale = 0.1
  const size = Math.max(width, height) * scale

  return (
    <group>
      <mesh>
        <boxGeometry
          args={[(size * width) / Math.max(width, height), (size * height) / Math.max(width, height), 0.2]}
        />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {`${width}x${height}x${channels}`}
      </Text>
    </group>
  )
}

// Flatten layer visualization
const FlattenLayer = ({ dimensions, color }: { dimensions: number[]; color: string }) => {
  const [flattenSize] = dimensions
  const height = (0.1 * flattenSize) / 100 // Scale height based on flattened size

  return (
    <group>
      <mesh>
        <boxGeometry args={[0.2, Math.min(height, 3), 0.2]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {flattenSize}
      </Text>
    </group>
  )
}

// Dense layer visualization
const DenseLayer = ({ neurons, color }: { neurons: number; color: string }) => {
  const neuronSpacing = 0.15
  const maxNeuronsToRender = 10
  const neuronsToRender = Math.min(neurons, maxNeuronsToRender)

  return (
    <group>
      {Array.from({ length: neuronsToRender }).map((_, i) => {
        const yPos = (i - (neuronsToRender - 1) / 2) * neuronSpacing
        return (
          <mesh key={i} position={[0, yPos, 0]}>
            <sphereGeometry args={[0.2, 16, 16]} />
            <meshStandardMaterial color={color} />
          </mesh>
        )
      })}

      {neurons > maxNeuronsToRender && (
        <Text position={[0, -2.5, 0]} fontSize={0.3} color="white" anchorX="center" anchorY="middle">
          {`...${neurons - maxNeuronsToRender} more`}
        </Text>
      )}

      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {neurons}
      </Text>
    </group>
  )
}

// Connection between CNN layers
const LayerConnection = ({
  startLayer,
  endLayer,
  distance,
  isActive,
}: {
  startLayer: any
  endLayer: any
  distance: number
  isActive: boolean
}) => {
  // Simplified connections between layers
  const connectorCount = 5 // Number of connectors to draw

  return (
    <group>
      {Array.from({ length: connectorCount }).map((_, i) => {
        const yOffset = (i - (connectorCount - 1) / 2) * 0.5
        return (
          <mesh key={i} position={[distance / 2, yOffset, 0]}>
            <cylinderGeometry args={[0.03, 0.03, distance, 8]} rotation={[0, Math.PI / 2, 0]} />
            <meshStandardMaterial color={isActive ? "#718096" : "#2d3748"} opacity={0.5} transparent={true} />
          </mesh>
        )
      })}
    </group>
  )
}

// RNN Input Layer
const RNNInputLayer = ({
  timeSteps,
  dimensions,
  color,
}: { timeSteps: number; dimensions: number[]; color: string }) => {
  const [seqLength, features] = dimensions
  const spacing = 1

  return (
    <group>
      {Array.from({ length: Math.min(timeSteps, seqLength) }).map((_, i) => {
        const xOffset = (i - Math.min(timeSteps, seqLength) / 2 + 0.5) * spacing
        return (
          <group key={i} position={[xOffset, 0, 0]}>
            <mesh>
              <boxGeometry args={[0.4, 0.8, 0.2]} />
              <meshStandardMaterial color={color} />
            </mesh>
            <Text position={[0, -1.2, 0]} fontSize={0.25} color="white" anchorX="center" anchorY="middle">
              {`t${i + 1}`}
            </Text>
          </group>
        )
      })}

      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {`${seqLength} x ${features}`}
      </Text>
    </group>
  )
}

// Recurrent Layer visualization
const RecurrentLayer = ({
  timeSteps,
  units,
  returnSequences,
  color,
}: {
  timeSteps: number
  units: number
  returnSequences: boolean
  color: string
}) => {
  const spacing = 1
  const displayTimeSteps = returnSequences ? timeSteps : 1

  return (
    <group>
      {Array.from({ length: displayTimeSteps }).map((_, i) => {
        const xOffset = returnSequences ? (i - displayTimeSteps / 2 + 0.5) * spacing : 0

        return (
          <group key={i} position={[xOffset, 0, 0]}>
            <mesh>
              <boxGeometry args={[0.8, 1.2, 0.3]} />
              <meshStandardMaterial color={color} />
            </mesh>

            {/* Recurrent connection */}
            {returnSequences && i < displayTimeSteps - 1 && (
              <mesh position={[spacing / 2, 0.8, 0]}>
                <cylinderGeometry args={[0.03, 0.03, spacing * 0.8, 8]} rotation={[0, Math.PI / 2, 0]} />
                <meshStandardMaterial color="#718096" opacity={0.7} transparent={true} />
              </mesh>
            )}

            {returnSequences && (
              <Text position={[0, -1.5, 0]} fontSize={0.25} color="white" anchorX="center" anchorY="middle">
                {`t${i + 1}`}
              </Text>
            )}
          </group>
        )
      })}

      <Text position={[0, -2, 0]} fontSize={0.4} color="white" anchorX="center" anchorY="middle">
        {units}
      </Text>
    </group>
  )
}

// Connection between RNN layers
const RNNLayerConnection = ({
  startLayer,
  endLayer,
  distance,
  isActive,
}: {
  startLayer: any
  endLayer: any
  distance: number
  isActive: boolean
}) => {
  const startHasSequence = startLayer.type === "input" || startLayer.returnSequences === true
  const endHasSequence = endLayer.type === "input" || endLayer.returnSequences === true

  if (startHasSequence && endHasSequence) {
    // Many-to-many connections
    return (
      <group>
        {Array.from({ length: 4 }).map((_, i) => {
          const startOffset = (i - 1.5) * 1
          const endOffset = (i - 1.5) * 1

          return (
            <Connection
              key={i}
              startPosition={[startOffset, 0, 0]}
              endPosition={[distance + endOffset, 0, 0]}
              isActive={isActive}
            />
          )
        })}
      </group>
    )
  } else if (startHasSequence && !endHasSequence) {
    // Many-to-one connections
    return (
      <group>
        {Array.from({ length: 4 }).map((_, i) => {
          const startOffset = (i - 1.5) * 1

          return (
            <Connection
              key={i}
              startPosition={[startOffset, 0, 0]}
              endPosition={[distance, 0, 0]}
              isActive={isActive}
            />
          )
        })}
      </group>
    )
  } else if (!startHasSequence && endHasSequence) {
    // One-to-many connections
    return (
      <group>
        {Array.from({ length: 4 }).map((_, i) => {
          const endOffset = (i - 1.5) * 1

          return (
            <Connection
              key={i}
              startPosition={[0, 0, 0]}
              endPosition={[distance + endOffset, 0, 0]}
              isActive={isActive}
            />
          )
        })}
      </group>
    )
  } else {
    // One-to-one connection
    return <Connection startPosition={[0, 0, 0]} endPosition={[distance, 0, 0]} isActive={isActive} />
  }
}

export default NetworkVisualization
