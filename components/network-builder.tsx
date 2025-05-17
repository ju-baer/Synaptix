"use client"

import { useRef, useState } from "react"
import { DndProvider, useDrag, useDrop } from "react-dnd"
import { HTML5Backend } from "react-dnd-html5-backend"
import type { NetworkConfig, LayerConfig, LayerType } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import {
  Plus,
  Minus,
  InfoIcon as InfoCircle,
  MoveVertical,
  Layers,
  SquareStack,
  ChevronsUpDown,
  RefreshCw,
  Maximize2,
  MinusSquare,
} from "lucide-react"

interface NetworkBuilderProps {
  config: NetworkConfig
  onChange: (config: NetworkConfig) => void
  onInfoClick: (topic: string) => void
}

const NetworkBuilder = ({ config, onChange, onInfoClick }: NetworkBuilderProps) => {
  const [activeTab, setActiveTab] = useState<string>("standard")

  const updateInputSize = (size: number) => {
    onChange({ ...config, inputSize: size })
  }

  const updateOutputSize = (size: number) => {
    onChange({ ...config, outputSize: size })
  }

  const updateHiddenLayer = (index: number, size: number) => {
    const newHiddenLayers = [...config.hiddenLayers]
    newHiddenLayers[index] = size
    onChange({ ...config, hiddenLayers: newHiddenLayers })
  }

  const addHiddenLayer = () => {
    onChange({
      ...config,
      hiddenLayers: [...config.hiddenLayers, 10],
    })
  }

  const removeHiddenLayer = (index: number) => {
    const newHiddenLayers = [...config.hiddenLayers]
    newHiddenLayers.splice(index, 1)
    onChange({ ...config, hiddenLayers: newHiddenLayers })
  }

  const updateActivation = (activation: string) => {
    onChange({ ...config, activation })
  }

  const updateInitializer = (initializer: string) => {
    onChange({ ...config, initializer })
  }

  const updateNetworkType = (type: string) => {
    // Create default layers based on network type
    let layers: LayerConfig[] = []

    if (type === "cnn") {
      layers = [
        { type: "input", config: { shape: [28, 28, 1] } },
        { type: "conv2d", config: { filters: 16, kernelSize: 3, activation: "relu" } },
        { type: "maxPooling2d", config: { poolSize: 2 } },
        { type: "conv2d", config: { filters: 32, kernelSize: 3, activation: "relu" } },
        { type: "maxPooling2d", config: { poolSize: 2 } },
        { type: "flatten", config: {} },
        { type: "dense", config: { units: 64, activation: "relu" } },
        { type: "dense", config: { units: 10, activation: "softmax" } },
      ]
    } else if (type === "rnn") {
      layers = [
        { type: "input", config: { shape: [10, 1] } }, // Sequence length 10, feature size 1
        { type: "lstm", config: { units: 32, returnSequences: false } },
        { type: "dense", config: { units: 16, activation: "relu" } },
        { type: "dense", config: { units: 1, activation: "linear" } },
      ]
    }

    onChange({
      ...config,
      networkType: type as "standard" | "cnn" | "rnn",
      layers: type !== "standard" ? layers : [],
    })

    setActiveTab(type)
  }

  const addLayer = (type: LayerType) => {
    let newLayer: LayerConfig

    switch (type) {
      case "conv2d":
        newLayer = { type, config: { filters: 16, kernelSize: 3, activation: "relu" } }
        break
      case "maxPooling2d":
        newLayer = { type, config: { poolSize: 2 } }
        break
      case "averagePooling2d":
        newLayer = { type, config: { poolSize: 2 } }
        break
      case "flatten":
        newLayer = { type, config: {} }
        break
      case "lstm":
        newLayer = { type, config: { units: 32, returnSequences: false } }
        break
      case "gru":
        newLayer = { type, config: { units: 32, returnSequences: false } }
        break
      case "simpleRNN":
        newLayer = { type, config: { units: 32, returnSequences: false } }
        break
      case "dense":
        newLayer = { type, config: { units: 16, activation: "relu" } }
        break
      case "dropout":
        newLayer = { type, config: { rate: 0.3 } }
        break
      default:
        newLayer = { type: "dense", config: { units: 16, activation: "relu" } }
    }

    onChange({
      ...config,
      layers: [...config.layers, newLayer],
    })
  }

  const updateLayer = (index: number, updatedLayer: LayerConfig) => {
    const newLayers = [...config.layers]
    newLayers[index] = updatedLayer
    onChange({
      ...config,
      layers: newLayers,
    })
  }

  const removeLayer = (index: number) => {
    // Don't remove input layer
    if (index === 0 && config.layers[0].type === "input") return

    const newLayers = [...config.layers]
    newLayers.splice(index, 1)
    onChange({
      ...config,
      layers: newLayers,
    })
  }

  const moveLayer = (dragIndex: number, hoverIndex: number) => {
    // Don't move input layer
    if (dragIndex === 0 && config.layers[0].type === "input") return
    if (hoverIndex === 0 && config.layers[0].type === "input") return

    const newLayers = [...config.layers]
    const temp = newLayers[dragIndex]
    newLayers[dragIndex] = newLayers[hoverIndex]
    newLayers[hoverIndex] = temp
    onChange({
      ...config,
      layers: newLayers,
    })
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">Network Architecture</h2>
        <Button size="sm" variant="ghost" onClick={() => onInfoClick("network_architecture")}>
          <InfoCircle className="h-4 w-4 mr-1" />
          Learn More
        </Button>
      </div>

      <Tabs defaultValue={config.networkType} onValueChange={updateNetworkType}>
        <TabsList className="grid w-full grid-cols-3 bg-gray-700">
          <TabsTrigger value="standard" className="data-[state=active]:bg-gray-600">
            <Layers className="h-4 w-4 mr-1" />
            Standard
          </TabsTrigger>
          <TabsTrigger value="cnn" className="data-[state=active]:bg-gray-600">
            <SquareStack className="h-4 w-4 mr-1" />
            CNN
          </TabsTrigger>
          <TabsTrigger value="rnn" className="data-[state=active]:bg-gray-600">
            <RefreshCw className="h-4 w-4 mr-1" />
            RNN
          </TabsTrigger>
        </TabsList>

        <TabsContent value="standard" className="space-y-3 pt-3">
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <label className="w-32">Input Size:</label>
              <div className="flex-1">
                <Input
                  type="number"
                  min={1}
                  value={config.inputSize}
                  onChange={(e) => updateInputSize(Number.parseInt(e.target.value) || 1)}
                  className="bg-gray-700 border-gray-600"
                />
              </div>
              <Button size="icon" variant="outline" onClick={() => onInfoClick("input_layer")}>
                <InfoCircle className="h-4 w-4" />
              </Button>
            </div>

            <DndProvider backend={HTML5Backend}>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label>Hidden Layers:</label>
                  <Button size="sm" variant="outline" onClick={addHiddenLayer}>
                    <Plus className="h-4 w-4 mr-1" />
                    Add Layer
                  </Button>
                </div>

                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {config.hiddenLayers.map((size, index) => (
                    <HiddenLayerItem
                      key={index}
                      index={index}
                      size={size}
                      onSizeChange={(size) => updateHiddenLayer(index, size)}
                      onRemove={() => removeHiddenLayer(index)}
                      onInfoClick={() => onInfoClick("hidden_layer")}
                      moveLayer={(dragIndex, hoverIndex) => {
                        const newLayers = [...config.hiddenLayers]
                        const temp = newLayers[dragIndex]
                        newLayers[dragIndex] = newLayers[hoverIndex]
                        newLayers[hoverIndex] = temp
                        onChange({ ...config, hiddenLayers: newLayers })
                      }}
                    />
                  ))}
                </div>
              </div>
            </DndProvider>

            <div className="flex items-center space-x-3">
              <label className="w-32">Output Size:</label>
              <div className="flex-1">
                <Input
                  type="number"
                  min={1}
                  value={config.outputSize}
                  onChange={(e) => updateOutputSize(Number.parseInt(e.target.value) || 1)}
                  className="bg-gray-700 border-gray-600"
                />
              </div>
              <Button size="icon" variant="outline" onClick={() => onInfoClick("output_layer")}>
                <InfoCircle className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="pt-3 border-t border-gray-700">
            <h3 className="text-lg font-medium mb-3">Network Parameters</h3>

            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <label className="w-32">Activation:</label>
                <Select value={config.activation} onValueChange={updateActivation}>
                  <SelectTrigger className="flex-1 bg-gray-700 border-gray-600">
                    <SelectValue placeholder="Select activation function" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    <SelectItem value="relu">ReLU</SelectItem>
                    <SelectItem value="sigmoid">Sigmoid</SelectItem>
                    <SelectItem value="tanh">Tanh</SelectItem>
                    <SelectItem value="leakyRelu">Leaky ReLU</SelectItem>
                  </SelectContent>
                </Select>
                <Button size="icon" variant="outline" onClick={() => onInfoClick("activation_functions")}>
                  <InfoCircle className="h-4 w-4" />
                </Button>
              </div>

              <div className="flex items-center space-x-3">
                <label className="w-32">Initializer:</label>
                <Select value={config.initializer} onValueChange={updateInitializer}>
                  <SelectTrigger className="flex-1 bg-gray-700 border-gray-600">
                    <SelectValue placeholder="Select weight initializer" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    <SelectItem value="glorotNormal">Glorot Normal</SelectItem>
                    <SelectItem value="glorotUniform">Glorot Uniform</SelectItem>
                    <SelectItem value="heNormal">He Normal</SelectItem>
                    <SelectItem value="heUniform">He Uniform</SelectItem>
                  </SelectContent>
                </Select>
                <Button size="icon" variant="outline" onClick={() => onInfoClick("weight_initializers")}>
                  <InfoCircle className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="cnn" className="space-y-3 pt-3">
          <AdvancedNetworkBuilder
            layers={config.layers}
            updateLayer={updateLayer}
            addLayer={addLayer}
            removeLayer={removeLayer}
            moveLayer={moveLayer}
            onInfoClick={onInfoClick}
            networkType="cnn"
          />
        </TabsContent>

        <TabsContent value="rnn" className="space-y-3 pt-3">
          <AdvancedNetworkBuilder
            layers={config.layers}
            updateLayer={updateLayer}
            addLayer={addLayer}
            removeLayer={removeLayer}
            moveLayer={moveLayer}
            onInfoClick={onInfoClick}
            networkType="rnn"
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}

interface AdvancedNetworkBuilderProps {
  layers: LayerConfig[]
  updateLayer: (index: number, layer: LayerConfig) => void
  addLayer: (type: LayerType) => void
  removeLayer: (index: number) => void
  moveLayer: (dragIndex: number, hoverIndex: number) => void
  onInfoClick: (topic: string) => void
  networkType: "cnn" | "rnn"
}

const AdvancedNetworkBuilder = ({
  layers,
  updateLayer,
  addLayer,
  removeLayer,
  moveLayer,
  onInfoClick,
  networkType,
}: AdvancedNetworkBuilderProps) => {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="text-base font-medium">Layer Configuration</h3>
        <Dialog>
          <DialogTrigger asChild>
            <Button size="sm" variant="outline">
              <Plus className="h-4 w-4 mr-1" />
              Add Layer
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-gray-800 border-gray-700">
            <DialogHeader>
              <DialogTitle>Add New Layer</DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-2">
              {networkType === "cnn" ? (
                <>
                  <Button variant="outline" onClick={() => addLayer("conv2d")}>
                    Convolutional 2D
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("maxPooling2d")}>
                    Max Pooling 2D
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("averagePooling2d")}>
                    Average Pooling 2D
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("flatten")}>
                    Flatten
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("dense")}>
                    Dense
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("dropout")}>
                    Dropout
                  </Button>
                </>
              ) : (
                <>
                  <Button variant="outline" onClick={() => addLayer("lstm")}>
                    LSTM
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("gru")}>
                    GRU
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("simpleRNN")}>
                    Simple RNN
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("dense")}>
                    Dense
                  </Button>
                  <Button variant="outline" onClick={() => addLayer("dropout")}>
                    Dropout
                  </Button>
                </>
              )}
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-2 max-h-72 overflow-y-auto">
        <DndProvider backend={HTML5Backend}>
          {layers.map((layer, index) => (
            <AdvancedLayerItem
              key={index}
              index={index}
              layer={layer}
              onUpdate={(updatedLayer) => updateLayer(index, updatedLayer)}
              onRemove={() => removeLayer(index)}
              onInfoClick={() => onInfoClick(layer.type)}
              moveLayer={moveLayer}
            />
          ))}
        </DndProvider>
      </div>

      <Button
        className="w-full"
        variant="outline"
        onClick={() => onInfoClick(networkType === "cnn" ? "cnn_architecture" : "rnn_architecture")}
      >
        <InfoCircle className="h-4 w-4 mr-1" />
        Learn About {networkType === "cnn" ? "CNN" : "RNN"} Architecture
      </Button>
    </div>
  )
}

interface AdvancedLayerItemProps {
  index: number
  layer: LayerConfig
  onUpdate: (layer: LayerConfig) => void
  onRemove: () => void
  onInfoClick: () => void
  moveLayer: (dragIndex: number, hoverIndex: number) => void
}

const AdvancedLayerItem = ({ index, layer, onUpdate, onRemove, onInfoClick, moveLayer }: AdvancedLayerItemProps) => {
  const ref = useRef<HTMLDivElement>(null)

  const [{ isDragging }, drag] = useDrag({
    type: "LAYER",
    item: { index },
    canDrag: layer.type !== "input" || index !== 0, // Don't allow dragging input layer
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  })

  const [, drop] = useDrop({
    accept: "LAYER",
    hover(item: { index: number }, monitor) {
      if (!ref.current) return
      const dragIndex = item.index
      const hoverIndex = index

      if (dragIndex === hoverIndex) return

      moveLayer(dragIndex, hoverIndex)
      item.index = hoverIndex
    },
  })

  drag(drop(ref))

  const isInput = layer.type === "input" && index === 0

  const renderLayerConfig = () => {
    switch (layer.type) {
      case "input":
        return (
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-xs">Width</label>
              <Input
                type="number"
                min={1}
                value={(layer.config.shape as number[])[0]}
                onChange={(e) => {
                  const shape = [...(layer.config.shape as number[])]
                  shape[0] = Number.parseInt(e.target.value) || 1
                  onUpdate({ ...layer, config: { ...layer.config, shape } })
                }}
                className="bg-gray-700 border-gray-600"
              />
            </div>
            <div>
              <label className="text-xs">Height</label>
              <Input
                type="number"
                min={1}
                value={(layer.config.shape as number[])[1]}
                onChange={(e) => {
                  const shape = [...(layer.config.shape as number[])]
                  shape[1] = Number.parseInt(e.target.value) || 1
                  onUpdate({ ...layer, config: { ...layer.config, shape } })
                }}
                className="bg-gray-700 border-gray-600"
              />
            </div>
            <div>
              <label className="text-xs">Channels</label>
              <Input
                type="number"
                min={1}
                value={(layer.config.shape as number[])[2]}
                onChange={(e) => {
                  const shape = [...(layer.config.shape as number[])]
                  shape[2] = Number.parseInt(e.target.value) || 1
                  onUpdate({ ...layer, config: { ...layer.config, shape } })
                }}
                className="bg-gray-700 border-gray-600"
              />
            </div>
          </div>
        )
      case "conv2d":
        return (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs">Filters</label>
              <Input
                type="number"
                min={1}
                value={layer.config.filters}
                onChange={(e) =>
                  onUpdate({ ...layer, config: { ...layer.config, filters: Number.parseInt(e.target.value) || 1 } })
                }
                className="bg-gray-700 border-gray-600"
              />
            </div>
            <div>
              <label className="text-xs">Kernel Size</label>
              <Input
                type="number"
                min={1}
                value={layer.config.kernelSize}
                onChange={(e) =>
                  onUpdate({ ...layer, config: { ...layer.config, kernelSize: Number.parseInt(e.target.value) || 1 } })
                }
                className="bg-gray-700 border-gray-600"
              />
            </div>
            <div className="col-span-2">
              <label className="text-xs">Activation</label>
              <Select
                value={layer.config.activation}
                onValueChange={(value) => onUpdate({ ...layer, config: { ...layer.config, activation: value } })}
              >
                <SelectTrigger className="bg-gray-700 border-gray-600">
                  <SelectValue placeholder="Select activation" />
                </SelectTrigger>
                <SelectContent className="bg-gray-700 border-gray-600">
                  <SelectItem value="relu">ReLU</SelectItem>
                  <SelectItem value="sigmoid">Sigmoid</SelectItem>
                  <SelectItem value="tanh">Tanh</SelectItem>
                  <SelectItem value="leakyRelu">Leaky ReLU</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )
      case "maxPooling2d":
      case "averagePooling2d":
        return (
          <div>
            <label className="text-xs">Pool Size</label>
            <Input
              type="number"
              min={1}
              value={layer.config.poolSize}
              onChange={(e) =>
                onUpdate({ ...layer, config: { ...layer.config, poolSize: Number.parseInt(e.target.value) || 1 } })
              }
              className="bg-gray-700 border-gray-600"
            />
          </div>
        )
      case "lstm":
      case "gru":
      case "simpleRNN":
        return (
          <div className="space-y-2">
            <div>
              <label className="text-xs">Units</label>
              <Input
                type="number"
                min={1}
                value={layer.config.units}
                onChange={(e) =>
                  onUpdate({ ...layer, config: { ...layer.config, units: Number.parseInt(e.target.value) || 1 } })
                }
                className="bg-gray-700 border-gray-600"
              />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">Return Sequences</label>
              <Button
                variant={layer.config.returnSequences ? "default" : "outline"}
                size="sm"
                onClick={() =>
                  onUpdate({ ...layer, config: { ...layer.config, returnSequences: !layer.config.returnSequences } })
                }
              >
                {layer.config.returnSequences ? "Enabled" : "Disabled"}
              </Button>
            </div>
          </div>
        )
      case "dense":
        return (
          <div className="space-y-2">
            <div>
              <label className="text-xs">Units</label>
              <Input
                type="number"
                min={1}
                value={layer.config.units}
                onChange={(e) =>
                  onUpdate({ ...layer, config: { ...layer.config, units: Number.parseInt(e.target.value) || 1 } })
                }
                className="bg-gray-700 border-gray-600"
              />
            </div>
            <div>
              <label className="text-xs">Activation</label>
              <Select
                value={layer.config.activation}
                onValueChange={(value) => onUpdate({ ...layer, config: { ...layer.config, activation: value } })}
              >
                <SelectTrigger className="bg-gray-700 border-gray-600">
                  <SelectValue placeholder="Select activation" />
                </SelectTrigger>
                <SelectContent className="bg-gray-700 border-gray-600">
                  <SelectItem value="relu">ReLU</SelectItem>
                  <SelectItem value="sigmoid">Sigmoid</SelectItem>
                  <SelectItem value="tanh">Tanh</SelectItem>
                  <SelectItem value="softmax">Softmax</SelectItem>
                  <SelectItem value="linear">Linear</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )
      case "dropout":
        return (
          <div>
            <label className="text-xs">Dropout Rate</label>
            <Input
              type="number"
              min={0}
              max={0.99}
              step={0.1}
              value={layer.config.rate}
              onChange={(e) => {
                const rate = Number.parseFloat(e.target.value)
                if (!isNaN(rate) && rate >= 0 && rate < 1) {
                  onUpdate({ ...layer, config: { ...layer.config, rate } })
                }
              }}
              className="bg-gray-700 border-gray-600"
            />
          </div>
        )
      default:
        return null
    }
  }

  const getLayerIcon = () => {
    switch (layer.type) {
      case "input":
        return <Layers className="h-4 w-4" />
      case "conv2d":
        return <SquareStack className="h-4 w-4" />
      case "maxPooling2d":
      case "averagePooling2d":
        return <Maximize2 className="h-4 w-4" />
      case "flatten":
        return <MinusSquare className="h-4 w-4" />
      case "lstm":
      case "gru":
      case "simpleRNN":
        return <RefreshCw className="h-4 w-4" />
      case "dense":
        return <ChevronsUpDown className="h-4 w-4" />
      default:
        return <Layers className="h-4 w-4" />
    }
  }

  return (
    <Card className={`${isDragging ? "opacity-50" : ""} bg-gray-700 border-gray-600`}>
      <CardContent className="p-3">
        <div className="space-y-2" ref={isInput ? null : ref}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {!isInput && <MoveVertical className="h-4 w-4 cursor-move text-gray-400" />}
              <div className="flex items-center space-x-1 font-medium">
                {getLayerIcon()}
                <span>
                  {layer.type.charAt(0).toUpperCase() + layer.type.slice(1)}{" "}
                  {layer.type !== "input" && layer.type !== "flatten" && `(${index})`}
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-1">
              <Button size="icon" variant="ghost" onClick={onInfoClick}>
                <InfoCircle className="h-4 w-4" />
              </Button>
              {!isInput && (
                <Button size="icon" variant="ghost" onClick={onRemove}>
                  <Minus className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
          {renderLayerConfig()}
        </div>
      </CardContent>
    </Card>
  )
}

interface HiddenLayerItemProps {
  index: number
  size: number
  onSizeChange: (size: number) => void
  onRemove: () => void
  onInfoClick: () => void
  moveLayer: (dragIndex: number, hoverIndex: number) => void
}

const ItemType = "HIDDEN_LAYER"

const HiddenLayerItem = ({ index, size, onSizeChange, onRemove, onInfoClick, moveLayer }: HiddenLayerItemProps) => {
  const ref = useRef<HTMLDivElement>(null)

  const [{ isDragging }, drag] = useDrag({
    type: ItemType,
    item: { index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  })

  const [, drop] = useDrop({
    accept: ItemType,
    hover(item: { index: number }, monitor) {
      if (!ref.current) {
        return
      }

      const dragIndex = item.index
      const hoverIndex = index

      if (dragIndex === hoverIndex) {
        return
      }

      moveLayer(dragIndex, hoverIndex)
      item.index = hoverIndex
    },
  })

  drag(drop(ref))

  return (
    <Card className={`${isDragging ? "opacity-50" : ""} bg-gray-700 border-gray-600`}>
      <CardContent className="p-2">
        <div className="flex items-center space-x-2" ref={ref}>
          <MoveVertical className="h-4 w-4 cursor-move text-gray-400" />
          <div className="flex-1 flex items-center space-x-2">
            <span className="text-sm text-gray-400">Layer {index + 1}:</span>
            <Input
              type="number"
              min={1}
              value={size}
              onChange={(e) => onSizeChange(Number.parseInt(e.target.value) || 1)}
              className="flex-1 bg-gray-700 border-gray-600"
            />
          </div>
          <Button size="icon" variant="ghost" onClick={onInfoClick}>
            <InfoCircle className="h-4 w-4" />
          </Button>
          <Button size="icon" variant="ghost" onClick={onRemove}>
            <Minus className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export default NetworkBuilder
