"use client"

import { useEffect, useState } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { educationalContent } from "@/lib/educational-content"

interface EducationalOverlayProps {
  topic: string
  onClose: () => void
}

const EducationalOverlay = ({ topic, onClose }: EducationalOverlayProps) => {
  const [content, setContent] = useState<{
    title: string
    description: string
    sections: {
      title: string
      content: string
    }[]
  } | null>(null)

  useEffect(() => {
    // Get content based on topic
    const topicContent = educationalContent[topic] || educationalContent["about"]
    setContent(topicContent)
  }, [topic])

  if (!content) return null

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>{content.title}</CardTitle>
            <CardDescription>{content.description}</CardDescription>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto">
          {content.sections.length > 1 ? (
            <Tabs defaultValue={content.sections[0].title.toLowerCase().replace(/\s+/g, "_")}>
              <TabsList className="w-full grid grid-cols-3">
                {content.sections.map((section) => (
                  <TabsTrigger key={section.title} value={section.title.toLowerCase().replace(/\s+/g, "_")}>
                    {section.title}
                  </TabsTrigger>
                ))}
              </TabsList>

              {content.sections.map((section) => (
                <TabsContent
                  key={section.title}
                  value={section.title.toLowerCase().replace(/\s+/g, "_")}
                  className="mt-4"
                >
                  <div className="prose prose-invert max-w-none">
                    <div dangerouslySetInnerHTML={{ __html: section.content }} />
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          ) : (
            <div className="prose prose-invert max-w-none">
              <div dangerouslySetInnerHTML={{ __html: content.sections[0].content }} />
            </div>
          )}
        </CardContent>

        <CardFooter className="border-t border-gray-700 pt-4">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}

export default EducationalOverlay
