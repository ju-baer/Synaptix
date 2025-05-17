"use client"

import type React from "react"

import { useState } from "react"
import type { DatasetType } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { InfoIcon as InfoCircle, Upload, Database } from "lucide-react"
import { builtinDatasets } from "@/lib/datasets"

interface DatasetManagerProps {
  onDatasetChange: (dataset: DatasetType | null) => void
  selectedDataset: DatasetType | null
  onInfoClick: (topic: string) => void
}

const DatasetManager = ({ onDatasetChange, selectedDataset, onInfoClick }: DatasetManagerProps) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    if (file) {
      setUploadedFile(file)
      // In a real app, we would parse the file here
      // For now, we'll just create a placeholder dataset
      onDatasetChange({
        id: "uploaded",
        name: file.name,
        type: "custom",
        features: [],
        labels: [],
        description: "Custom uploaded dataset",
      })
    }
  }

  const selectBuiltinDataset = (datasetId: string) => {
    const dataset = builtinDatasets.find((d) => d.id === datasetId)
    if (dataset) {
      onDatasetChange(dataset)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">Dataset Selection</h2>
        <Button size="sm" variant="ghost" onClick={() => onInfoClick("datasets")}>
          <InfoCircle className="h-4 w-4 mr-1" />
          Learn More
        </Button>
      </div>

      <Tabs defaultValue="builtin">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="builtin">Built-in Datasets</TabsTrigger>
          <TabsTrigger value="upload">Upload Dataset</TabsTrigger>
        </TabsList>

        <TabsContent value="builtin" className="space-y-3 pt-3">
          <div className="grid grid-cols-2 gap-3">
            {builtinDatasets.map((dataset) => (
              <Card
                key={dataset.id}
                className={`cursor-pointer transition-all ${
                  selectedDataset?.id === dataset.id ? "border-blue-500 bg-blue-500/10" : "hover:border-gray-600"
                }`}
                onClick={() => selectBuiltinDataset(dataset.id)}
              >
                <CardContent className="p-3">
                  <div className="flex flex-col items-center text-center">
                    <Database className="h-8 w-8 mb-2" />
                    <h3 className="font-medium">{dataset.name}</h3>
                    <p className="text-xs text-gray-400 mt-1">{dataset.description}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="upload" className="space-y-3 pt-3">
          <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center">
            <div className="flex flex-col items-center">
              <Upload className="h-10 w-10 mb-3 text-gray-400" />
              <h3 className="text-lg font-medium mb-1">Upload Your Dataset</h3>
              <p className="text-sm text-gray-400 mb-4">Supported formats: CSV, JSON</p>

              <div className="flex flex-col items-center">
                <Button asChild>
                  <label>
                    <input type="file" accept=".csv,.json" className="hidden" onChange={handleFileUpload} />
                    Browse Files
                  </label>
                </Button>

                {uploadedFile && <p className="mt-3 text-sm text-blue-400">Selected: {uploadedFile.name}</p>}
              </div>
            </div>
          </div>

          <Button size="sm" variant="outline" className="w-full" onClick={() => onInfoClick("dataset_format")}>
            <InfoCircle className="h-4 w-4 mr-1" />
            Dataset Format Guidelines
          </Button>
        </TabsContent>
      </Tabs>

      {selectedDataset && (
        <div className="pt-3 border-t border-gray-700">
          <h3 className="text-lg font-medium mb-2">Selected Dataset: {selectedDataset.name}</h3>
          <p className="text-sm text-gray-400 mb-3">{selectedDataset.description}</p>

          <Button variant="outline" size="sm" onClick={() => onInfoClick("dataset_visualization")} className="w-full">
            <InfoCircle className="h-4 w-4 mr-1" />
            Visualize Dataset
          </Button>
        </div>
      )}
    </div>
  )
}

export default DatasetManager
