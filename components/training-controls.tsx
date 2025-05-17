"use client"

import type { TrainingConfig } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { InfoIcon as InfoCircle, Play, Pause, Square, StepForward } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface TrainingControlsProps {
  config: TrainingConfig
  onChange: (config: TrainingConfig) => void
  isTraining: boolean
  isPaused: boolean
  onStart: () => void
  onPauseResume: () => void
  onStop: () => void
  currentEpoch: number
  loss: number[]
  accuracy: number[]
  onInfoClick: (topic: string) => void
}

const TrainingControls = ({
  config,
  onChange,
  isTraining,
  isPaused,
  onStart,
  onPauseResume,
  onStop,
  currentEpoch,
  loss,
  accuracy,
  onInfoClick,
}: TrainingControlsProps) => {
  const updateLearningRate = (value: number) => {
    onChange({ ...config, learningRate: value })
  }

  const updateEpochs = (value: number) => {
    onChange({ ...config, epochs: value })
  }

  const updateBatchSize = (value: number) => {
    onChange({ ...config, batchSize: value })
  }

  const updateOptimizer = (value: string) => {
    onChange({ ...config, optimizer: value })
  }

  const updateLossFunction = (value: string) => {
    onChange({ ...config, lossFunction: value })
  }

  // Prepare chart data
  const chartData = loss.map((lossValue, index) => ({
    epoch: index + 1,
    loss: lossValue,
    accuracy: accuracy[index] || 0,
  }))

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">Training Parameters</h2>
        <Button size="sm" variant="ghost" onClick={() => onInfoClick("training_parameters")}>
          <InfoCircle className="h-4 w-4 mr-1" />
          Learn More
        </Button>
      </div>

      <div className="space-y-3">
        <div className="space-y-1">
          <div className="flex justify-between">
            <label className="text-sm">Learning Rate: {config.learningRate}</label>
            <Button size="icon" variant="ghost" className="h-5 w-5" onClick={() => onInfoClick("learning_rate")}>
              <InfoCircle className="h-3 w-3" />
            </Button>
          </div>
          <Slider
            value={[config.learningRate]}
            min={0.0001}
            max={0.1}
            step={0.0001}
            onValueChange={(value) => updateLearningRate(value[0])}
          />
        </div>

        <div className="flex items-center space-x-3">
          <label className="w-32">Epochs:</label>
          <div className="flex-1">
            <Input
              type="number"
              min={1}
              value={config.epochs}
              onChange={(e) => updateEpochs(Number.parseInt(e.target.value) || 1)}
            />
          </div>
          <Button size="icon" variant="outline" onClick={() => onInfoClick("epochs")}>
            <InfoCircle className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center space-x-3">
          <label className="w-32">Batch Size:</label>
          <div className="flex-1">
            <Input
              type="number"
              min={1}
              value={config.batchSize}
              onChange={(e) => updateBatchSize(Number.parseInt(e.target.value) || 1)}
            />
          </div>
          <Button size="icon" variant="outline" onClick={() => onInfoClick("batch_size")}>
            <InfoCircle className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center space-x-3">
          <label className="w-32">Optimizer:</label>
          <Select value={config.optimizer} onValueChange={updateOptimizer}>
            <SelectTrigger className="flex-1">
              <SelectValue placeholder="Select optimizer" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sgd">SGD</SelectItem>
              <SelectItem value="adam">Adam</SelectItem>
              <SelectItem value="rmsprop">RMSprop</SelectItem>
              <SelectItem value="adagrad">Adagrad</SelectItem>
            </SelectContent>
          </Select>
          <Button size="icon" variant="outline" onClick={() => onInfoClick("optimizers")}>
            <InfoCircle className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center space-x-3">
          <label className="w-32">Loss Function:</label>
          <Select value={config.lossFunction} onValueChange={updateLossFunction}>
            <SelectTrigger className="flex-1">
              <SelectValue placeholder="Select loss function" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="meanSquaredError">Mean Squared Error</SelectItem>
              <SelectItem value="categoricalCrossentropy">Categorical Crossentropy</SelectItem>
              <SelectItem value="binaryCrossentropy">Binary Crossentropy</SelectItem>
            </SelectContent>
          </Select>
          <Button size="icon" variant="outline" onClick={() => onInfoClick("loss_functions")}>
            <InfoCircle className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="pt-3 border-t border-gray-700">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-lg font-medium">Training Controls</h3>
          {isTraining && (
            <div className="text-sm text-blue-400">
              Epoch: {currentEpoch} / {config.epochs}
            </div>
          )}
        </div>

        <div className="flex space-x-2 mb-4">
          {!isTraining ? (
            <Button className="flex-1" onClick={onStart}>
              <Play className="h-4 w-4 mr-1" />
              Start Training
            </Button>
          ) : (
            <>
              <Button className="flex-1" variant={isPaused ? "default" : "outline"} onClick={onPauseResume}>
                {isPaused ? (
                  <>
                    <Play className="h-4 w-4 mr-1" />
                    Resume
                  </>
                ) : (
                  <>
                    <Pause className="h-4 w-4 mr-1" />
                    Pause
                  </>
                )}
              </Button>

              <Button
                variant="outline"
                onClick={() => {
                  /* Step forward logic */
                }}
                disabled={!isPaused}
              >
                <StepForward className="h-4 w-4" />
              </Button>

              <Button variant="destructive" onClick={onStop}>
                <Square className="h-4 w-4 mr-1" />
                Stop
              </Button>
            </>
          )}
        </div>

        {chartData.length > 0 && (
          <div className="h-40 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "insideBottomRight", offset: -5 }}
                  stroke="#a0aec0"
                />
                <YAxis stroke="#a0aec0" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#2d3748",
                    borderColor: "#4a5568",
                    color: "#fff",
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#f56565" activeDot={{ r: 8 }} />
                <Line type="monotone" dataKey="accuracy" stroke="#4299e1" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

export default TrainingControls
