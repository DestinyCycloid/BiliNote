import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, CheckCircle2, Info } from 'lucide-react'
import toast from 'react-hot-toast'
import request from '@/utils/request'

interface Transcriber {
  value: string
  label: string
  description: string
}

const Transcriber = () => {
  const [transcribers, setTranscribers] = useState<Transcriber[]>([])
  const [currentTranscriber, setCurrentTranscriber] = useState<string>('')
  const [selectedTranscriber, setSelectedTranscriber] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  // 加载转写器列表
  useEffect(() => {
    loadTranscribers()
  }, [])

  const loadTranscribers = async () => {
    setLoading(true)
    try {
      const response = await request.get('/transcribers')
      if (response && response.data) {
        setTranscribers(response.data.transcribers)
        setCurrentTranscriber(response.data.current)
        setSelectedTranscriber(response.data.current)
      }
    } catch (error) {
      console.error('加载转写器列表失败:', error)
      toast.error('加载转写器列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (selectedTranscriber === currentTranscriber) {
      toast.success('当前已是该转写器')
      return
    }

    setSaving(true)
    try {
      const response = await request.post('/transcriber/set', {
        transcriber_type: selectedTranscriber,
      })
      
      if (response && response.data) {
        toast.success('转写器设置成功！重启后生效')
        setCurrentTranscriber(selectedTranscriber)
      }
    } catch (error) {
      console.error('设置转写器失败:', error)
      toast.error('设置转写器失败')
    } finally {
      setSaving(false)
    }
  }

  const selectedInfo = transcribers.find(t => t.value === selectedTranscriber)

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="container mx-auto max-w-4xl p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">转写器设置</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          选择音频转写引擎，不同引擎有不同的特点和性能
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>选择转写器</CardTitle>
          <CardDescription>
            当前使用：<span className="font-semibold">{transcribers.find(t => t.value === currentTranscriber)?.label || currentTranscriber}</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">转写引擎</label>
            <Select value={selectedTranscriber} onValueChange={setSelectedTranscriber}>
              <SelectTrigger>
                <SelectValue placeholder="选择转写器" />
              </SelectTrigger>
              <SelectContent>
                {transcribers.map(transcriber => (
                  <SelectItem key={transcriber.value} value={transcriber.value}>
                    <div className="flex flex-col">
                      <span className="font-medium">{transcriber.label}</span>
                      <span className="text-xs text-muted-foreground">{transcriber.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {selectedInfo && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription>
                <div className="space-y-1">
                  <p className="font-semibold">{selectedInfo.label}</p>
                  <p className="text-sm">{selectedInfo.description}</p>
                </div>
              </AlertDescription>
            </Alert>
          )}

          <div className="flex gap-2">
            <Button
              onClick={handleSave}
              disabled={saving || selectedTranscriber === currentTranscriber}
              className="flex-1"
            >
              {saving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  保存中...
                </>
              ) : selectedTranscriber === currentTranscriber ? (
                <>
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                  当前使用
                </>
              ) : (
                '保存设置'
              )}
            </Button>
          </div>

          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription className="text-sm">
              <p className="font-semibold mb-2">转写器说明：</p>
              <ul className="space-y-1 text-xs">
                <li>• <strong>Faster Whisper</strong>: 通用选择，支持多语言</li>
                <li>• <strong>Paraformer-streaming</strong>: 中文优化，支持流式转写（推荐）</li>
                <li>• <strong>Fun-ASR-Nano</strong>: 轻量级，支持31种语言</li>
                <li>• <strong>Deepgram/Groq</strong>: 云端API，需要配置API Key</li>
                <li>• <strong>必剪/快手</strong>: 官方转写服务</li>
              </ul>
              <p className="mt-2 text-amber-600">⚠️ 修改后需要重启应用才能生效</p>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </div>
  )
}

export default Transcriber
