"use client"

import { useState } from "react"
import NetworkBuilder from "@/components/network-builder"
import NetworkVisualization from "@/components/network-visualization"
import TrainingControls from "@/components/training-controls"
import DatasetManager from "@/components/dataset-manager"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { InfoIcon as InfoCircle } from "lucide-react"
import type { NetworkConfig, DatasetType, TrainingConfig } from "@/lib/types"
import { defaultNetworkConfig, defaultTrainingConfig } from "@/lib/defaults"
import { createModel } from "@/lib/model"
import EducationalOverlay from "@/components/educational-overlay"

export default function SynaptixApp() {
  const [networkConfig, setNetworkConfig] = useState<NetworkConfig>(defaultNetworkConfig)
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(defaultTrainingConfig)
  const [dataset, setDataset] = useState<DatasetType | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [loss, setLoss] = useState<number[]>([])
  const [accuracy, setAccuracy] = useState<number[]>([])
  const [showOverlay, setShowOverlay] = useState(false)
  const [overlayTopic, setOverlayTopic] = useState("")

  const handleStartTraining = async () => {
    if (!dataset) {
      alert("Please select a dataset first")
      return
    }

    setIsTraining(true)
    setIsPaused(false)
    setCurrentEpoch(0)
    setLoss([])
    setAccuracy([])

    const model = createModel(networkConfig)

    // Training loop would be implemented here with TensorFlow.js
    // This is a placeholder for the actual implementation
  }

  const handlePauseResume = () => {
    setIsPaused(!isPaused)
  }

  const handleStop = () => {
    setIsTraining(false)
    setIsPaused(false)
  }

  const handleShowOverlay = (topic: string) => {
    setOverlayTopic(topic)
    setShowOverlay(true)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <header className="border-b border-gray-700 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-blue-400">Synaptix</h1>
          <div className="flex gap-4">
            <Button variant="outline" onClick={() => handleShowOverlay("about")}>
              About Synaptix
            </Button>
            <Button variant="outline" onClick={() => handleShowOverlay("help")}>
              Help
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg shadow-xl p-4 h-[600px] relative">
              <NetworkVisualization
                networkConfig={networkConfig}
                isTraining={isTraining}
                isPaused={isPaused}
                currentEpoch={currentEpoch}
              />
              <Button
                size="sm"
                variant="ghost"
                className="absolute top-2 right-2"
                onClick={() => handleShowOverlay("visualization")}
              >
                <InfoCircle className="h-4 w-4 mr-1" />
                Learn about this visualization
              </Button>
            </div>
          </div>

          <div className="space-y-6">
            <Tabs defaultValue="builder" className="bg-gray-800 rounded-lg shadow-xl">
              <TabsList className="w-full grid grid-cols-3 bg-gray-700">
                <TabsTrigger value="builder" className="data-[state=active]:bg-gray-600">
                  Network
                </TabsTrigger>
                <TabsTrigger value="dataset" className="data-[state=active]:bg-gray-600">
                  Dataset
                </TabsTrigger>
                <TabsTrigger value="training" className="data-[state=active]:bg-gray-600">
                  Training
                </TabsTrigger>
              </TabsList>

              <TabsContent value="builder" className="p-4">
                <NetworkBuilder config={networkConfig} onChange={setNetworkConfig} onInfoClick={handleShowOverlay} />
              </TabsContent>

              <TabsContent value="dataset" className="p-4">
                <DatasetManager
                  onDatasetChange={setDataset}
                  selectedDataset={dataset}
                  onInfoClick={handleShowOverlay}
                />
              </TabsContent>

              <TabsContent value="training" className="p-4">
                <TrainingControls
                  config={trainingConfig}
                  onChange={setTrainingConfig}
                  isTraining={isTraining}
                  isPaused={isPaused}
                  onStart={handleStartTraining}
                  onPauseResume={handlePauseResume}
                  onStop={handleStop}
                  currentEpoch={currentEpoch}
                  loss={loss}
                  accuracy={accuracy}
                  onInfoClick={handleShowOverlay}
                />
              </TabsContent>
            </Tabs>

            <div className="bg-gray-800 rounded-lg shadow-xl p-4">
              <h2 className="text-xl font-semibold mb-3">Export & Share</h2>
              <div className="space-y-3">
                <Button className="w-full" variant="outline">
                  Export Configuration
                </Button>
                <Button className="w-full" variant="outline">
                  Generate Share Link
                </Button>
                <Button className="w-full" variant="outline">
                  Export Training Results
                </Button>
              </div>
            </div>
          </div>
        </div>
      </main>

      {showOverlay && <EducationalOverlay topic={overlayTopic} onClose={() => setShowOverlay(false)} />}
    </div>
  )
}
