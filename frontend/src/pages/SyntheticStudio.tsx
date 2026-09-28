import React, { useState, useEffect, useRef, useCallback } from "react"
import {
  Upload,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Car,
  Sliders,
  Camera,
  CheckCircle2,
  Clock,
  ShieldCheck,
  Search,
  Download,
  AlertCircle,
  FileVideo,
  Bell,
  BellOff,
  ShieldAlert,
  AlertTriangle,
  UserX,
  Ban,
  Eye,
  X,
  Volume2,
  VolumeX,
  Check,
  Radio,
  Tv,
  Settings,
  RefreshCw,
  Cpu,
  Activity,
  Plus,
  Trash2,
  ChevronDown,
  ChevronUp,
  Tag
} from "lucide-react"
import { API_BASE, getEvidenceUrl } from "../services/api"

// ─── Interfaces ─────────────────────────────────────────────────────────────

interface VideoItem {
  id: string
  filename: string
  path: string
  display_name: string
  width: number
  height: number
  fps: number
  frame_count: number
  duration_sec: number
  size_mb: number
  is_upload: boolean
  thumbnail_url: string
  video_url?: string
  blob_url?: string
  playable_url?: string
}

interface VehicleTrack {
  track_id: string
  type: string
  conf: number
  box_pixel?: [number, number, number, number]
  box_norm: {
    x: number
    y: number
    w: number
    h: number
  }
  plate_number: string | null
  plate_conf: number
  has_plate: boolean
  is_watchlist_match: boolean
  watchlist_category?: string | null
  watchlist_priority?: string | null
  vehicle_model?: string | null
  case_number?: string | null
  description?: string | null
  snapshot_url?: string | null
  plate_snapshot_url?: string | null
  status_text: string
  frames_seen: number
}

interface AlertItem {
  alert_id: string
  alert_type: string
  vehicle_number: string
  vehicle_model: string
  camera_name: string
  detection_time: string
  anpr_confidence: number
  vehicle_detection_confidence: number
  evidence_frame: string | null
  plate_frame: string | null
  watchlist_category: string
  alert_priority: string
  status: string
  case_number?: string
  description?: string
  timestamp_iso: string
  video: string
  track_id: string
  acknowledged_by?: string
  acknowledged_at?: string
}

interface DetectionStats {
  vehicles_detected: number
  plates_detected: number
  ocr_results: number
  watchlist_matches: number
  alerts: number
}

interface EventItem {
  id: string
  time: string
  text: string
  details?: string
  type: "vehicle" | "plate" | "ocr" | "match" | "alert" | "system"
}

interface WatchlistEntry {
  watchlist_id?: string
  registration_number: string
  category: string
  priority: string
  model?: string
  description?: string
  case_number?: string
  jurisdiction?: string
}

// 8 Pre-loaded synthetic camera feeds from Gujarat Police dataset
const DEFAULT_SYNTHETIC_VIDEOS: VideoItem[] = [
  {
    id: "1.mp4",
    filename: "1.mp4",
    path: "Synthetic Dataset/1.mp4",
    display_name: "CAM-01 • Chimanbhai Bridge North",
    width: 1920,
    height: 1080,
    fps: 30,
    frame_count: 900,
    duration_sec: 30,
    size_mb: 28.2,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/1.mp4",
    video_url: "/api/synthetic/video-file/1.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "2.mp4",
    filename: "2.mp4",
    path: "Synthetic Dataset/2.mp4",
    display_name: "CAM-02 • Janpath Intersection",
    width: 1920,
    height: 1080,
    fps: 25,
    frame_count: 497,
    duration_sec: 19.9,
    size_mb: 7.9,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/2.mp4",
    video_url: "/api/synthetic/video-file/2.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "3.mp4",
    filename: "3.mp4",
    path: "Synthetic Dataset/3.mp4",
    display_name: "CAM-03 • ONGC Circle South",
    width: 1920,
    height: 1080,
    fps: 30,
    frame_count: 750,
    duration_sec: 25,
    size_mb: 13.4,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/3.mp4",
    video_url: "/api/synthetic/video-file/3.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "4.mp4",
    filename: "4.mp4",
    path: "Synthetic Dataset/4.mp4",
    display_name: "CAM-04 • Paldi Crossroad",
    width: 1920,
    height: 1080,
    fps: 30,
    frame_count: 1200,
    duration_sec: 40,
    size_mb: 68.0,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/4.mp4",
    video_url: "/api/synthetic/video-file/4.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "5.mp4",
    filename: "5.mp4",
    path: "Synthetic Dataset/5.mp4",
    display_name: "CAM-05 • SG Highway Toll Plaza",
    width: 1920,
    height: 1080,
    fps: 30,
    frame_count: 900,
    duration_sec: 30,
    size_mb: 44.5,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/5.mp4",
    video_url: "/api/synthetic/video-file/5.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "6.mp4",
    filename: "6.mp4",
    path: "Synthetic Dataset/6.mp4",
    display_name: "CAM-06 • Ashram Road Junction",
    width: 1920,
    height: 1080,
    fps: 25,
    frame_count: 600,
    duration_sec: 24,
    size_mb: 9.8,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/6.mp4",
    video_url: "/api/synthetic/video-file/6.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "7.mp4",
    filename: "7.mp4",
    path: "Synthetic Dataset/7.mp4",
    display_name: "CAM-07 • Gandhinagar Ring Road",
    width: 1920,
    height: 1080,
    fps: 30,
    frame_count: 750,
    duration_sec: 25,
    size_mb: 19.6,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/7.mp4",
    video_url: "/api/synthetic/video-file/7.mp4",
    playable_url: "/sample_cctv.mp4"
  },
  {
    id: "8.mp4",
    filename: "8.mp4",
    path: "Synthetic Dataset/8.mp4",
    display_name: "CAM-08 • Sindhu Bhavan Corridor",
    width: 1920,
    height: 1080,
    fps: 25,
    frame_count: 500,
    duration_sec: 20,
    size_mb: 8.0,
    is_upload: false,
    thumbnail_url: "/api/synthetic/thumbnail/8.mp4",
    video_url: "/api/synthetic/video-file/8.mp4",
    playable_url: "/sample_cctv.mp4"
  }
]

// Category visual styling helper
function getCategoryBadge(category: string) {
  switch (category) {
    case "STOLEN_VEHICLE":
      return { bg: "bg-red-950/90", border: "border-red-600", text: "text-red-300", icon: <ShieldAlert className="w-3.5 h-3.5" />, label: "STOLEN VEHICLE" }
    case "WANTED_PERSON":
      return { bg: "bg-orange-950/90", border: "border-orange-500", text: "text-orange-300", icon: <UserX className="w-3.5 h-3.5" />, label: "WANTED PERSON" }
    case "MISSING_PERSON_VEHICLE":
      return { bg: "bg-yellow-950/90", border: "border-yellow-500", text: "text-yellow-300", icon: <Eye className="w-3.5 h-3.5" />, label: "MISSING PERSON" }
    case "BLACKLISTED_VEHICLE":
      return { bg: "bg-purple-950/90", border: "border-purple-600", text: "text-purple-300", icon: <Ban className="w-3.5 h-3.5" />, label: "BLACKLISTED" }
    case "SUSPECT_VEHICLE":
      return { bg: "bg-amber-950/90", border: "border-amber-500", text: "text-amber-300", icon: <AlertTriangle className="w-3.5 h-3.5" />, label: "SUSPECT VEHICLE" }
    default:
      return { bg: "bg-slate-900/90", border: "border-slate-600", text: "text-slate-300", icon: <AlertCircle className="w-3.5 h-3.5" />, label: category ? category.replace("_", " ") : "ALERT" }
  }
}

export const SyntheticStudio: React.FC = () => {
  // Video selection & player states
  const [videos, setVideos] = useState<VideoItem[]>(DEFAULT_SYNTHETIC_VIDEOS)
  const [selectedVideo, setSelectedVideo] = useState<string>("1.mp4")
  const [isPlaying, setIsPlaying] = useState<boolean>(true)
  const [isMuted, setIsMuted] = useState<boolean>(true)
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0)
  const [isUploading, setIsUploading] = useState<boolean>(false)
  const [uploadNotice, setUploadNotice] = useState<string | null>(null)

  // Confidence Thresholds (Adjustable by operator)
  const [vehicleConf, setVehicleConf] = useState<number>(0.35)
  const [plateConf, setPlateConf] = useState<number>(0.25)
  const [ocrConf, setOcrConf] = useState<number>(0.35)
  const [anprConf, setAnprConf] = useState<number>(0.50)
  const [detectorInterval, setDetectorInterval] = useState<number>(3) // process frame every N ticks

  // AI Pipeline Dynamic Output States
  const [activeVehicles, setActiveVehicles] = useState<VehicleTrack[]>([])
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [stats, setStats] = useState<DetectionStats>({
    vehicles_detected: 0,
    plates_detected: 0,
    ocr_results: 0,
    watchlist_matches: 0,
    alerts: 0
  })
  const [eventHistory, setEventHistory] = useState<EventItem[]>([
    {
      id: "EVT-INIT",
      time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
      text: "System initialized",
      details: "AstraSight Dynamic AI Engine ready",
      type: "system"
    }
  ])

  // UI Tabs & Panels
  const [activeTab, setActiveTab] = useState<"detections" | "alerts">("detections")
  const [searchQuery, setSearchQuery] = useState<string>("")
  const [showThresholds, setShowThresholds] = useState<boolean>(false)
  const [showWatchlistModal, setShowWatchlistModal] = useState<boolean>(false)
  const [showDebugPanel, setShowDebugPanel] = useState<boolean>(true)
  const [engineMode, setEngineMode] = useState<"live" | "synthetic">("live")

  // Watchlist Management
  const [watchlistEntries, setWatchlistEntries] = useState<WatchlistEntry[]>([])
  const [newPlateInput, setNewPlateInput] = useState<string>("")
  const [newModelInput, setNewModelInput] = useState<string>("")
  const [newCatInput, setNewCatInput] = useState<string>("STOLEN_VEHICLE")
  const [newPrioInput, setNewPrioInput] = useState<string>("CRITICAL")
  const [newCaseInput, setNewCaseInput] = useState<string>("")

  // Debug / Latency Metrics
  const [debugLatencyMs, setDebugLatencyMs] = useState<number>(38.5)
  const [debugFps, setDebugFps] = useState<number>(3.6)
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)

  // Refs
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const eventLogEndRef = useRef<HTMLDivElement | null>(null)
  const isAnalyzingRef = useRef<boolean>(false)
  const frameTickRef = useRef<number>(0)
  const lastAnalyzedTimeRef = useRef<number>(0)

  // Active video item
  const activeVideo = videos.find((v) => v.filename === selectedVideo) || videos[0]
  const activePlayableUrl =
    activeVideo?.blob_url ||
    (activeVideo?.video_url ? getEvidenceUrl(activeVideo.video_url) : undefined) ||
    activeVideo?.playable_url ||
    "/sample_cctv.mp4"

  // Load available videos from backend
  useEffect(() => {
    fetchVideos()
    fetchWatchlist()
  }, [])

  const fetchVideos = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/synthetic/videos`)
      if (res.ok) {
        const data: VideoItem[] = await res.json()
        if (data && data.length > 0) {
          setVideos(data)
        }
      }
    } catch {
      // Backend offline; use default 8 feeds
    }
  }

  const fetchWatchlist = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/synthetic/watchlist`)
      if (res.ok) {
        const data = await res.json()
        if (data.watchlist) {
          setWatchlistEntries(data.watchlist)
        }
      }
    } catch {
      // Fallback
    }
  }

  // ─── Live Frame Capture & Transmission Pipeline ────────────────────────────
  const analyzeCurrentFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    if (video.paused || video.ended || video.readyState < 2) return
    if (isAnalyzingRef.current) return // Prevent overlapping requests

    const curTime = video.currentTime
    // Throttle analysis: process at most once per 280ms
    if (Math.abs(curTime - lastAnalyzedTimeRef.current) < 0.25) return

    isAnalyzingRef.current = true
    setIsAnalyzing(true)
    lastAnalyzedTimeRef.current = curTime

    const tStart = performance.now()

    try {
      const canvas = canvasRef.current
      const ctx = canvas.getContext("2d")
      if (!ctx) {
        isAnalyzingRef.current = false
        setIsAnalyzing(false)
        return
      }

      // Draw current video frame preserving high detail for number plate OCR
      const srcW = video.videoWidth || 1280
      const srcH = video.videoHeight || 720
      const targetW = Math.min(srcW, 960)
      const targetH = Math.round(targetW * (srcH / srcW))
      canvas.width = targetW
      canvas.height = targetH
      ctx.drawImage(video, 0, 0, targetW, targetH)

      const b64Data = canvas.toDataURL("image/jpeg", 0.82)

      const payload = {
        video_id: selectedVideo,
        timestamp_sec: curTime,
        image_base64: b64Data,
        vehicle_conf: vehicleConf,
        plate_conf: plateConf,
        ocr_conf: ocrConf,
        anpr_conf: anprConf,
        camera_name: activeVideo?.display_name || "CAM-01"
      }

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000)

      const res = await fetch(`${API_BASE}/api/synthetic/analyze-frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal
      })
      clearTimeout(timeoutId)


      if (res.ok) {
        const data = await res.json()
        const tElapsed = Math.round(performance.now() - tStart)
        setDebugLatencyMs(tElapsed)
        setDebugFps(Number((1000 / Math.max(tElapsed, 200)).toFixed(1)))

        // 1. Dynamic Vehicles
        if (Array.isArray(data.vehicles)) {
          setActiveVehicles(data.vehicles)
        }

        // 2. Dynamic Stats
        if (data.stats) {
          setStats(data.stats)
        }

        // 3. Dynamic Events
        if (Array.isArray(data.new_events) && data.new_events.length > 0) {
          setEventHistory((prev) => [...prev, ...data.new_events].slice(-60))
        }

        // 4. Dynamic Alerts
        if (data.new_alert) {
          setAlerts((prev) => {
            const exists = prev.some((a) => a.alert_id === data.new_alert.alert_id)
            if (exists) return prev
            return [data.new_alert, ...prev]
          })
          setActiveTab("alerts")
        }
      }
    } catch (err) {
      console.warn("Live frame analysis error:", err)
    } finally {
      isAnalyzingRef.current = false
      setIsAnalyzing(false)
    }
  }, [
    selectedVideo,
    vehicleConf,
    plateConf,
    ocrConf,
    anprConf,
    activeVideo
  ])

  // Periodic frame loop driven by video time updates & interval
  useEffect(() => {
    if (!isPlaying || engineMode !== "live") return

    const intervalId = setInterval(() => {
      frameTickRef.current += 1
      if (frameTickRef.current % detectorInterval === 0) {
        analyzeCurrentFrame()
      }
    }, 100)

    return () => clearInterval(intervalId)
  }, [isPlaying, engineMode, detectorInterval, analyzeCurrentFrame])

  // Play / Pause toggle
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
        setIsPlaying(false)
      } else {
        videoRef.current.play().catch(() => {})
        setIsPlaying(true)
      }
    } else {
      setIsPlaying(!isPlaying)
    }
  }

  // Restart playback and reset tracker state
  const restartPlayback = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play().catch(() => {})
      setIsPlaying(true)
      setCurrentTimeSec(0)
    }
    setActiveVehicles([])
    fetch(`${API_BASE}/api/synthetic/reset-session?video=${encodeURIComponent(selectedVideo)}`, {
      method: "POST"
    }).catch(() => {})
    setEventHistory((prev) => [
      ...prev,
      {
        id: `EVT-${Date.now()}`,
        time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
        text: "Playback restarted",
        details: "Trackers & OCR buffers reset",
        type: "system"
      }
    ])
  }

  // Switch video stream
  const handleSelectVideo = (filename: string) => {
    fetch(`${API_BASE}/api/synthetic/reset-session?video=${encodeURIComponent(filename)}`, {
      method: "POST"
    }).catch(() => {})
    setSelectedVideo(filename)
    setIsPlaying(true)
    setCurrentTimeSec(0)
    setActiveVehicles([])
    setAlerts([])
    setStats({
      vehicles_detected: 0,
      plates_detected: 0,
      ocr_results: 0,
      watchlist_matches: 0,
      alerts: 0
    })
    setEventHistory([
      {
        id: `EVT-SW-${Date.now()}`,
        time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
        text: `Switched stream: ${filename}`,
        details: "Live pipeline initialized for new feed",
        type: "system"
      }
    ])
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play().catch(() => {})
    }
  }

  // In-Browser Custom Video Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsUploading(true)
    setUploadNotice(null)

    // Instant local blob URL for instant playback
    const localBlobUrl = URL.createObjectURL(file)
    const customVid: VideoItem = {
      id: `upload_${Date.now()}`,
      filename: file.name,
      path: file.name,
      display_name: `Custom Upload • ${file.name.length > 18 ? file.name.substring(0, 15) + "..." : file.name}`,
      width: 1920,
      height: 1080,
      fps: 30,
      frame_count: 450,
      duration_sec: 15,
      size_mb: parseFloat((file.size / (1024 * 1024)).toFixed(1)),
      is_upload: true,
      thumbnail_url: "",
      blob_url: localBlobUrl,
      playable_url: localBlobUrl
    }

    setVideos((prev) => [customVid, ...prev.filter((v) => v.filename !== file.name)])
    setSelectedVideo(file.name)
    setIsPlaying(true)
    setCurrentTimeSec(0)
    setActiveVehicles([])
    setAlerts([])
    setIsUploading(false)
    setUploadNotice(`Custom Video "${file.name}" loaded. Dynamic AI Frame Pipeline active.`)

    setEventHistory([
      {
        id: `EVT-UP-${Date.now()}`,
        time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
        text: `Loaded user video: ${file.name}`,
        details: "Extracting frames for dynamic YOLO + ANPR detection",
        type: "system"
      }
    ])

    // Upload to backend in background
    try {
      const formData = new FormData()
      formData.append("file", file)
      fetch(`${API_BASE}/api/synthetic/upload`, { method: "POST", body: formData }).catch(() => {})
    } catch {
      // Handled
    }

    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  // Acknowledge alert
  const handleAcknowledgeAlert = (alertId: string) => {
    setAlerts((prev) =>
      prev.map((a) =>
        a.alert_id === alertId
          ? {
              ...a,
              status: "ACKNOWLEDGED",
              acknowledged_by: "Operator (K. Patel - GP-4082)",
              acknowledged_at: new Date().toLocaleTimeString("en-GB", { hour12: false })
            }
          : a
      )
    )
    fetch(`${API_BASE}/api/synthetic/alerts/${alertId}/acknowledge?operator=K. Patel`, {
      method: "POST"
    }).catch(() => {})
  }

  // Watchlist: Add entry
  const handleAddWatchlist = async () => {
    if (!newPlateInput.trim()) return
    const cleanReg = newPlateInput.trim().toUpperCase().replace(/[^A-Z0-9]/g, "")
    const newEntry: WatchlistEntry = {
      registration_number: cleanReg,
      model: newModelInput.trim() || "Unknown Model",
      category: newCatInput,
      priority: newPrioInput,
      case_number: newCaseInput.trim() || `FIR-GJ-${cleanReg.slice(0, 4)}`,
      description: `Watchlist target (${newCatInput.replace("_", " ")}). High surveillance priority.`
    }

    try {
      const res = await fetch(`${API_BASE}/api/synthetic/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newEntry)
      })
      if (res.ok) {
        setWatchlistEntries((prev) => [newEntry, ...prev.filter((w) => w.registration_number !== cleanReg)])
        setNewPlateInput("")
        setNewModelInput("")
        setNewCaseInput("")
        setUploadNotice(`Watchlist target ${cleanReg} added to surveillance database.`)
      }
    } catch {
      // Fallback local update
      setWatchlistEntries((prev) => [newEntry, ...prev])
    }
  }

  // Watchlist: Delete entry
  const handleDeleteWatchlist = async (regNumber: string) => {
    try {
      await fetch(`${API_BASE}/api/synthetic/watchlist/${encodeURIComponent(regNumber)}`, {
        method: "DELETE"
      })
      setWatchlistEntries((prev) => prev.filter((w) => w.registration_number !== regNumber))
    } catch {
      // Fallback
    }
  }

  // Export report
  const handleExportReport = () => {
    const reportData = {
      video: selectedVideo,
      analyzed_at: new Date().toISOString(),
      stats,
      active_vehicles: activeVehicles,
      alerts,
      event_history: eventHistory
    }
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `cctv_ai_report_${selectedVideo.replace(".mp4", "")}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const unackAlertsCount = alerts.filter((a) => a.status !== "ACKNOWLEDGED").length

  return (
    <div className="space-y-6">
      {/* Hidden Offscreen Canvas for Real-Time Frame Capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-police-950 p-5 rounded-xl border border-police-800 shadow-xl">
        <div className="flex items-center gap-3">
          <span className="p-2.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            <Sparkles className="w-5 h-5" />
          </span>
          <div>
            <h1 className="text-xl font-bold text-white tracking-wide flex items-center gap-3">
              Live AI Studio &amp; ANPR Testing
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-mono font-medium">
                YOLOv8 + ByteTrack + EasyOCR
              </span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Gujarat Police Autonomous Surveillance • Dynamic Frame Analysis &amp; Watchlist Matching
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 text-xs flex-wrap">
          {/* Watchlist Manager Button */}
          <button
            onClick={() => setShowWatchlistModal(true)}
            className="px-3 py-2 bg-police-900 hover:bg-police-800 text-cyan-300 rounded-lg border border-police-700 flex items-center gap-2 transition-colors cursor-pointer"
          >
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold">Watchlist Database</span>
            <span className="px-1.5 py-0.2 bg-cyan-950 border border-cyan-700 text-cyan-200 rounded text-[10px] font-mono">
              {watchlistEntries.length}
            </span>
          </button>

          {/* Thresholds Toggle Button */}
          <button
            onClick={() => setShowThresholds(!showThresholds)}
            className={`px-3 py-2 rounded-lg border flex items-center gap-2 transition-colors cursor-pointer ${
              showThresholds
                ? "bg-cyan-950 text-cyan-300 border-cyan-500"
                : "bg-police-900 hover:bg-police-800 text-slate-300 border-police-700"
            }`}
          >
            <Sliders className="w-4 h-4 text-police-400" />
            <span className="font-medium">Thresholds</span>
            {showThresholds ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {/* Active Alerts Badge */}
          {unackAlertsCount > 0 && (
            <div className="px-3 py-2 bg-red-950/80 rounded-lg border border-red-700 flex items-center gap-2 animate-pulse">
              <ShieldAlert className="w-4 h-4 text-red-400" />
              <div>
                <span className="text-red-400 block text-[9px] font-bold uppercase">ALERTS</span>
                <span className="font-semibold text-red-300">{unackAlertsCount} Unack</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Active Watchlist Match Full-Width Banner ─────────────────────── */}
      {unackAlertsCount > 0 && (
        <div className="flex items-center gap-3 px-4 py-3 bg-red-950/80 border border-red-700 rounded-xl shadow-lg shadow-red-950/50 animate-pulse">
          <span className="relative flex h-3 w-3 shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
          </span>
          <ShieldAlert className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm font-bold text-red-200 flex-1">
            CRITICAL WATCHLIST MATCH — {unackAlertsCount} vehicle alert{unackAlertsCount > 1 ? "s" : ""} detected on current feed. Immediate law enforcement verification required.
          </p>
          <button
            onClick={() => setActiveTab("alerts")}
            className="px-3.5 py-1.5 bg-red-700 hover:bg-red-600 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
          >
            Inspect Alerts ({unackAlertsCount})
          </button>
        </div>
      )}

      {/* ── Upload Notification Toast ──────────────────────────────────── */}
      {uploadNotice && (
        <div className="flex items-center justify-between px-4 py-2.5 bg-cyan-950/70 border border-cyan-700 rounded-xl text-xs text-cyan-200">
          <span className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-cyan-400" />
            {uploadNotice}
          </span>
          <button onClick={() => setUploadNotice(null)} className="text-slate-400 hover:text-white cursor-pointer">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ── Confidence Thresholds Drawer (Expandable) ─────────────────── */}
      {showThresholds && (
        <div className="bg-police-950 p-4 rounded-xl border border-police-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              Real-Time AI Model Confidence Thresholds
            </span>
            <span className="text-[11px] text-slate-400">
              Adjust thresholds live — dynamically filtered per video frame
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 pt-1">
            {/* 1. Vehicle Conf */}
            <div className="bg-police-900/80 p-3 rounded-lg border border-police-800 space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Vehicle Detection</span>
                <span className="font-mono text-cyan-400 font-bold">{Math.round(vehicleConf * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.85"
                step="0.05"
                value={vehicleConf}
                onChange={(e) => setVehicleConf(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 block">YOLOv8 car/truck/bus filter</span>
            </div>

            {/* 2. Plate Conf */}
            <div className="bg-police-900/80 p-3 rounded-lg border border-police-800 space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Plate Localization</span>
                <span className="font-mono text-cyan-400 font-bold">{Math.round(plateConf * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.05"
                value={plateConf}
                onChange={(e) => setPlateConf(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 block">License plate bounding box</span>
            </div>

            {/* 3. OCR Conf */}
            <div className="bg-police-900/80 p-3 rounded-lg border border-police-800 space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">OCR Character Conf</span>
                <span className="font-mono text-cyan-400 font-bold">{Math.round(ocrConf * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.15"
                max="0.85"
                step="0.05"
                value={ocrConf}
                onChange={(e) => setOcrConf(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 block">EasyOCR token acceptance</span>
            </div>

            {/* 4. ANPR Consensus Conf */}
            <div className="bg-police-900/80 p-3 rounded-lg border border-police-800 space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Final ANPR Consensus</span>
                <span className="font-mono text-emerald-400 font-bold">{Math.round(anprConf * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.25"
                max="0.90"
                step="0.05"
                value={anprConf}
                onChange={(e) => setAnprConf(Number(e.target.value))}
                className="w-full accent-emerald-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500 block">Multi-frame agreement gate</span>
            </div>
          </div>
        </div>
      )}

      {/* ── Detection Status Section Bar ──────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {/* Vehicles Detected */}
        <div className="p-3 bg-police-950/90 rounded-xl border border-police-800/80 flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Car className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
              Vehicles Detected
            </span>
            <span className="text-lg font-black text-white font-mono">{stats.vehicles_detected}</span>
          </div>
        </div>

        {/* Plates Detected */}
        <div className="p-3 bg-police-950/90 rounded-xl border border-police-800/80 flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Tag className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
              Plates Detected
            </span>
            <span className="text-lg font-black text-white font-mono">{stats.plates_detected}</span>
          </div>
        </div>

        {/* OCR Results */}
        <div className="p-3 bg-police-950/90 rounded-xl border border-police-800/80 flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Eye className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
              OCR Results
            </span>
            <span className="text-lg font-black text-white font-mono">{stats.ocr_results}</span>
          </div>
        </div>

        {/* Watchlist Matches */}
        <div className="p-3 bg-police-950/90 rounded-xl border border-police-800/80 flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
              Watchlist Matches
            </span>
            <span className="text-lg font-black text-amber-400 font-mono">{stats.watchlist_matches}</span>
          </div>
        </div>

        {/* Alerts Generated */}
        <div className="p-3 bg-police-950/90 rounded-xl border border-police-800/80 flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20">
            <Bell className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
              Active Alerts
            </span>
            <span className="text-lg font-black text-red-400 font-mono">{alerts.length}</span>
          </div>
        </div>
      </div>

      {/* ── Video Selector Shelf ────────────────────────────────────────── */}
      <div className="bg-police-950/80 p-4 rounded-xl border border-police-800">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-police-300 uppercase tracking-wider flex items-center gap-2">
            <FileVideo className="w-3.5 h-3.5 text-cyan-400" />
            Select CCTV Feed or Upload Video
          </span>
          <span className="text-[11px] text-slate-400">
            Detections are dynamically extracted from actual video frames
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-3">
          {/* Upload Button */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group cursor-pointer flex flex-col items-center justify-center p-3 rounded-lg border-2 border-dashed border-cyan-500/60 hover:border-cyan-400 bg-cyan-950/20 hover:bg-cyan-950/40 transition-all text-center min-h-[105px]"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="video/mp4,video/avi,video/quicktime,video/mkv,video/webm"
              className="hidden"
            />
            <Upload className="w-6 h-6 text-cyan-400 group-hover:scale-110 transition-transform mb-1.5" />
            <span className="text-xs font-semibold text-white">Upload MP4</span>
            <span className="text-[10px] text-cyan-300 mt-0.5">{isUploading ? "Reading..." : "Custom Video"}</span>
          </div>

          {/* Feeds */}
          {videos.map((vid) => {
            const isSelected = vid.filename === selectedVideo
            return (
              <div
                key={vid.id}
                onClick={() => handleSelectVideo(vid.filename)}
                className={`relative group cursor-pointer rounded-lg overflow-hidden border transition-all ${
                  isSelected
                    ? "border-cyan-400 ring-2 ring-cyan-500/40 shadow-lg shadow-cyan-950"
                    : "border-police-800 hover:border-police-600 bg-police-900/60"
                }`}
              >
                <div className="aspect-video w-full bg-slate-900 relative flex items-center justify-center">
                  {vid.thumbnail_url ? (
                    <img
                      src={getEvidenceUrl(vid.thumbnail_url)}
                      alt={vid.display_name}
                      onError={(e) => {
                        const target = e.currentTarget
                        const localThumb = `/thumbnails/${vid.filename.replace(".mp4", "")}.jpg`
                        if (!target.src.endsWith(localThumb)) {
                          target.src = localThumb
                        } else {
                          ;(e.target as HTMLElement).style.display = "none"
                        }
                      }}
                    />

                  ) : (
                    <div className="flex flex-col items-center justify-center p-2 text-slate-500">
                      <Camera className="w-5 h-5 text-police-500 mb-1" />
                      <span className="text-[9px] font-mono text-slate-400">{vid.filename}</span>
                    </div>
                  )}
                  {isSelected && (
                    <span className="absolute top-1 left-1 px-1.5 py-0.5 bg-cyan-500 text-[8px] font-bold text-black rounded uppercase tracking-wider">
                      ACTIVE
                    </span>
                  )}
                  {vid.is_upload && (
                    <span className="absolute top-1 right-1 px-1.5 py-0.5 bg-amber-500 text-[8px] font-bold text-black rounded uppercase tracking-wider">
                      UPLOAD
                    </span>
                  )}
                  <span className="absolute bottom-1 right-1 px-1 bg-black/80 text-[9px] font-mono text-slate-200 rounded">
                    {vid.duration_sec}s
                  </span>
                </div>
                <div className="p-2 bg-police-950">
                  <p className="text-[11px] font-medium text-white truncate">{vid.display_name}</p>
                  <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                    {vid.width}x{vid.height} • {vid.fps}fps
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Main Workstation Grid ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 8 Cols: Video Player + Overlays + Controls + Event History */}
        <div className="lg:col-span-8 space-y-4">
          <div className="relative rounded-xl overflow-hidden border border-cyan-500/40 bg-black shadow-2xl shadow-cyan-950/40">
            <div className="aspect-video w-full relative flex items-center justify-center bg-slate-950">
              {/* Native HTML5 Video Element */}
              <video
                key={selectedVideo}
                ref={videoRef}
                src={activePlayableUrl}
                autoPlay
                loop
                muted={isMuted}
                playsInline
                crossOrigin="anonymous"
                className="w-full h-full object-contain"
                onLoadedData={() => {
                  if (isPlaying && videoRef.current) {
                    videoRef.current.play().catch(() => {})
                  }
                }}
                onTimeUpdate={() => {
                  if (videoRef.current) {
                    setCurrentTimeSec(videoRef.current.currentTime)
                  }
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onError={(e) => {
                  const target = e.currentTarget
                  if (!target.src.endsWith("/sample_cctv.mp4")) {
                    target.src = "/sample_cctv.mp4"
                  }
                }}
              />


              {/* ── Real-Time Dynamic AI SVG HUD Overlay ──────────────── */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1000 562.5">
                {activeVehicles.map((veh) => {
                  // Normalized coordinates (0 to 1) converted to 1000x562.5 viewBox
                  const bx = veh.box_norm.x * 1000
                  const by = veh.box_norm.y * 562.5
                  const bw = veh.box_norm.w * 1000
                  const bh = veh.box_norm.h * 562.5

                  const isMatch = Boolean(veh.is_watchlist_match)
                  // Normal vehicle = THIN GREEN (#10b981), Watchlist match = THIN RED (#ef4444)
                  const boxStroke = isMatch ? "#ef4444" : "#10b981"
                  const cornerLen = Math.min(14, Math.max(6, bw * 0.15))

                  return (
                    <g key={veh.track_id} className="transition-all duration-100">
                      {/* Corner Bracket Reticles */}
                      {/* Top-Left */}
                      <path
                        d={`M ${bx} ${by + cornerLen} L ${bx} ${by} L ${bx + cornerLen} ${by}`}
                        stroke={boxStroke}
                        strokeWidth="2.5"
                        fill="none"
                      />
                      {/* Top-Right */}
                      <path
                        d={`M ${bx + bw - cornerLen} ${by} L ${bx + bw} ${by} L ${bx + bw} ${by + cornerLen}`}
                        stroke={boxStroke}
                        strokeWidth="2.5"
                        fill="none"
                      />
                      {/* Bottom-Left */}
                      <path
                        d={`M ${bx} ${by + bh - cornerLen} L ${bx} ${by + bh} L ${bx + cornerLen} ${by + bh}`}
                        stroke={boxStroke}
                        strokeWidth="2.5"
                        fill="none"
                      />
                      {/* Bottom-Right */}
                      <path
                        d={`M ${bx + bw - cornerLen} ${by + bh} L ${bx + bw} ${by + bh} L ${bx + bw} ${by + bh - cornerLen}`}
                        stroke={boxStroke}
                        strokeWidth="2.5"
                        fill="none"
                      />

                      {/* Thin Bounding Box */}
                      <rect
                        x={bx}
                        y={by}
                        width={bw}
                        height={bh}
                        fill={isMatch ? "rgba(239, 68, 68, 0.08)" : "rgba(16, 185, 129, 0.04)"}
                        stroke={boxStroke}
                        strokeWidth="1.2"
                        strokeDasharray={isMatch ? "none" : "3 1.5"}
                      />

                      {/* Top Vehicle Tracking Tag: Vehicle ID + Type + Conf */}
                      <rect
                        x={bx}
                        y={Math.max(by - 22, 6)}
                        width={130}
                        height={18}
                        rx="3"
                        fill={isMatch ? "#7f1d1d" : "#022c22"}
                        stroke={boxStroke}
                        strokeWidth="1"
                      />
                      <text
                        x={bx + 6}
                        y={Math.max(by - 9, 19)}
                        fill="#ffffff"
                        fontSize="10"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {veh.track_id} • {veh.type} {Math.round(veh.conf * 100)}%
                      </text>

                      {/* Bottom Plate Lock Tag */}
                      <rect
                        x={bx}
                        y={by + bh + 3}
                        width={145}
                        height={20}
                        rx="3"
                        fill={isMatch ? "#991b1b" : "#0f172a"}
                        stroke={isMatch ? "#ef4444" : veh.plate_number ? "#10b981" : "#475569"}
                        strokeWidth="1.2"
                      />
                      <text
                        x={bx + 6}
                        y={by + bh + 17}
                        fill={isMatch ? "#fecaca" : veh.plate_number ? "#6ee7b7" : "#94a3b8"}
                        fontSize="11"
                        fontFamily="monospace"
                        fontWeight="900"
                        letterSpacing="1"
                      >
                        {veh.plate_number ? `IND ${veh.plate_number}` : veh.has_plate ? "PLATE SCANNING..." : "NO READABLE PLATE"}
                      </text>

                      {/* Watchlist Alert Tag (If Matched) */}
                      {isMatch && (
                        <g>
                          <rect
                            x={bx}
                            y={Math.max(by - 44, 26)}
                            width={180}
                            height={19}
                            rx="3"
                            fill="#dc2626"
                          />
                          <text
                            x={bx + 6}
                            y={Math.max(by - 31, 39)}
                            fill="#ffffff"
                            fontSize="9.5"
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            🚨 WATCHLIST: {veh.watchlist_category?.replace("_", " ")}
                          </text>
                        </g>
                      )}
                    </g>
                  )
                })}
              </svg>

              {/* No Vehicles Detected Banner in HUD */}
              {activeVehicles.length === 0 && (
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-black/70 border border-slate-700/80 rounded-full text-slate-400 text-xs font-mono pointer-events-none flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
                  Analyzing Video Frames • No reliable vehicle detection
                </div>
              )}

              {/* LIVE Frame Analysis Badge Top-Left */}
              <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
                <span className="flex h-2.5 w-2.5 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
                </span>
                <span className="px-2 py-0.5 rounded bg-black/85 text-[10px] font-mono font-bold text-cyan-300 tracking-wider border border-cyan-500/40 uppercase">
                  {engineMode === "live" ? "LIVE FRAME ANALYSIS • REAL-TIME AI" : "SYNTHETIC DEMO MODE"}
                </span>
                {isAnalyzing && (
                  <span className="px-1.5 py-0.5 bg-emerald-950/80 border border-emerald-500/60 text-emerald-300 text-[9px] font-mono rounded">
                    {debugLatencyMs}ms
                  </span>
                )}
              </div>

              {/* Active Alerts Badge Top-Right */}
              {unackAlertsCount > 0 && (
                <div className="absolute top-3 right-3 flex items-center gap-1.5 px-2.5 py-1 bg-red-900/90 border border-red-600 rounded pointer-events-none animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-red-400 animate-ping inline-flex shrink-0"></span>
                  <span className="text-[10px] font-bold text-red-200 font-mono uppercase tracking-wider">
                    {unackAlertsCount} WATCHLIST MATCH{unackAlertsCount > 1 ? "ES" : ""}
                  </span>
                </div>
              )}
            </div>

            {/* Video Player Controls Bar */}
            <div className="p-3.5 bg-police-950 border-t border-police-800 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <button
                  onClick={togglePlay}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-police-800 hover:bg-police-700 text-white text-xs font-medium transition-colors cursor-pointer"
                >
                  {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <button
                  onClick={restartPlayback}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-police-800 hover:bg-police-700 text-white text-xs font-medium transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Restart
                </button>
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className="p-1.5 rounded-lg bg-police-800 hover:bg-police-700 text-slate-300 hover:text-white transition-colors cursor-pointer"
                  title={isMuted ? "Unmute" : "Mute"}
                >
                  {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5 text-cyan-400" />}
                </button>
              </div>

              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  <span className="font-mono text-slate-300 text-xs">
                    {Math.floor(currentTimeSec / 60)}:
                    {(currentTimeSec % 60).toFixed(1).padStart(4, "0")}s
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                  <Car className="w-3.5 h-3.5 text-emerald-400" />
                  <span>On Screen:</span>
                  <span className="font-mono font-bold text-white">{activeVehicles.length}</span>
                </div>
                <button
                  onClick={handleExportReport}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-police-900 hover:bg-police-800 text-cyan-300 text-xs border border-police-700 transition-colors cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  Report
                </button>
              </div>
            </div>
          </div>

          {/* ── Event History Log Section Below Video ──────────────────────── */}
          <div className="bg-police-950 p-4 rounded-xl border border-police-800 space-y-2.5 shadow-lg">
            <div className="flex items-center justify-between border-b border-police-800 pb-2">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                Live Surveillance Event History
              </span>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-400 font-mono">
                  {eventHistory.length} events logged
                </span>
                <button
                  onClick={() => setEventHistory([])}
                  className="text-[10px] text-slate-400 hover:text-white px-2 py-0.5 rounded bg-police-900 border border-police-800 cursor-pointer"
                >
                  Clear
                </button>
              </div>
            </div>

            {/* Scrollable Event Log Window */}
            <div className="h-36 overflow-y-auto font-mono text-xs space-y-1.5 pr-2">
              {eventHistory.length === 0 ? (
                <div className="text-slate-600 text-center py-6 text-xs">
                  Awaiting detection events from video frames...
                </div>
              ) : (
                eventHistory.map((ev) => {
                  let badgeStyle = "text-slate-400 border-slate-700 bg-slate-900"
                  if (ev.type === "vehicle") badgeStyle = "text-emerald-400 border-emerald-800 bg-emerald-950"
                  if (ev.type === "plate") badgeStyle = "text-cyan-400 border-cyan-800 bg-cyan-950"
                  if (ev.type === "ocr") badgeStyle = "text-amber-300 border-amber-800 bg-amber-950"
                  if (ev.type === "match") badgeStyle = "text-red-300 border-red-700 bg-red-950"
                  if (ev.type === "alert") badgeStyle = "text-red-200 border-red-600 bg-red-900 font-bold"

                  return (
                    <div
                      key={ev.id}
                      className="flex items-start gap-2.5 py-1 px-2 rounded bg-police-900/60 border border-police-800/60 hover:border-police-700 transition-colors"
                    >
                      <span className="text-slate-500 shrink-0 text-[11px] font-mono">{ev.time}</span>
                      <span className="text-slate-600 shrink-0">—</span>
                      <span className={`px-1.5 py-0.2 rounded text-[10px] border ${badgeStyle} shrink-0`}>
                        {ev.text}
                      </span>
                      {ev.details && (
                        <span className="text-slate-300 truncate text-[11px]">{ev.details}</span>
                      )}
                    </div>
                  )
                })
              )}
              <div ref={eventLogEndRef} />
            </div>
          </div>

          {/* ── Demo / Debug Panel ────────────────────────────────────────── */}
          <div className="bg-police-950/80 p-3.5 rounded-xl border border-police-800/80 text-xs text-slate-400">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span className="font-semibold text-white">Surveillance Engine Diagnostics</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-700 font-mono">
                  LIVE FRAME ANALYSIS
                </span>
              </div>
              <button
                onClick={() => setEngineMode(engineMode === "live" ? "synthetic" : "live")}
                className="text-[10px] text-cyan-300 hover:underline cursor-pointer"
              >
                Switch to {engineMode === "live" ? "Synthetic Fallback" : "Live Frame Analysis"}
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-2.5 pt-2.5 border-t border-police-800/60 text-[11px]">
              <div>
                <span className="text-slate-500 block">Inference Latency:</span>
                <span className="font-mono text-cyan-300 font-bold">{debugLatencyMs} ms</span>
              </div>
              <div>
                <span className="text-slate-500 block">Pipeline Throughput:</span>
                <span className="font-mono text-emerald-300 font-bold">{debugFps} FPS</span>
              </div>
              <div>
                <span className="text-slate-500 block">Model Stack:</span>
                <span className="font-mono text-slate-300">YOLOv8n + LP-YOLO + EasyOCR</span>
              </div>
              <div>
                <span className="text-slate-500 block">Tracking Engine:</span>
                <span className="font-mono text-slate-300">ByteTrack / Centroid IoU</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right 4 Cols: Tabbed Detections & Alerts Panel */}
        <div className="lg:col-span-4 bg-police-950 rounded-xl border border-police-800 flex flex-col h-[740px] shadow-xl overflow-hidden">
          {/* Tab Header */}
          <div className="border-b border-police-800">
            <div className="flex">
              <button
                onClick={() => setActiveTab("detections")}
                className={`flex-1 flex items-center justify-center gap-2 py-3 text-xs font-semibold transition-colors cursor-pointer ${
                  activeTab === "detections"
                    ? "text-cyan-300 border-b-2 border-cyan-400 bg-cyan-950/20"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                Vehicles ({activeVehicles.length})
              </button>
              <button
                onClick={() => setActiveTab("alerts")}
                className={`flex-1 flex items-center justify-center gap-2 py-3 text-xs font-semibold transition-colors relative cursor-pointer ${
                  activeTab === "alerts"
                    ? "text-red-300 border-b-2 border-red-500 bg-red-950/20"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {unackAlertsCount > 0 ? (
                  <Bell className="w-3.5 h-3.5 text-red-400 animate-pulse" />
                ) : (
                  <BellOff className="w-3.5 h-3.5" />
                )}
                Alerts ({alerts.length})
                {unackAlertsCount > 0 && (
                  <span className="absolute top-2 right-6 w-4 h-4 bg-red-600 text-white text-[9px] font-bold rounded-full flex items-center justify-center">
                    {unackAlertsCount}
                  </span>
                )}
              </button>
            </div>

            {/* Sub-header Filter */}
            <div className="px-4 py-2 flex items-center justify-between">
              {activeTab === "detections" ? (
                <div className="relative flex-1">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search Vehicle ID / Plate..."
                    className="w-full bg-police-900 border border-police-700/80 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 outline-none focus:border-cyan-500"
                  />
                </div>
              ) : (
                <span className="text-[11px] text-slate-400 font-mono">
                  {unackAlertsCount > 0
                    ? `${unackAlertsCount} unacknowledged — operator dispatch required`
                    : "No unacknowledged alerts on this feed"}
                </span>
              )}
            </div>
          </div>

          {/* ── DETECTIONS TAB CONTENT ── */}
          {activeTab === "detections" && (
            <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
              {activeVehicles.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
                  <Car className="w-8 h-8 mb-2 opacity-40 text-cyan-400" />
                  <p className="text-xs">No vehicles in current frame</p>
                  <p className="text-[10px] mt-1 text-slate-600">
                    Vehicles dynamically appear here as frames are processed
                  </p>
                </div>
              ) : (
                activeVehicles
                  .filter((v) => {
                    if (!searchQuery) return true
                    const q = searchQuery.toUpperCase()
                    return (
                      v.track_id.toUpperCase().includes(q) ||
                      (v.plate_number && v.plate_number.toUpperCase().includes(q)) ||
                      v.type.toUpperCase().includes(q)
                    )
                  })
                  .map((veh) => {
                    const isMatch = Boolean(veh.is_watchlist_match)
                    const catBadge = isMatch && veh.watchlist_category ? getCategoryBadge(veh.watchlist_category) : null

                    return (
                      <div
                        key={veh.track_id}
                        className={`p-3 rounded-lg border transition-all space-y-2 ${
                          isMatch
                            ? "bg-red-950/40 border-red-700/80 hover:border-red-500"
                            : "bg-police-900/70 border-police-800/90 hover:border-cyan-500/50"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span
                              className={`px-2 py-0.5 rounded text-xs font-bold font-mono border ${
                                isMatch
                                  ? "bg-red-950 text-red-300 border-red-700"
                                  : "bg-emerald-950/90 text-emerald-300 border-emerald-800"
                              }`}
                            >
                              {veh.track_id} • {veh.type}
                            </span>
                            <span className="text-[10px] text-slate-400 font-mono">
                              {Math.round(veh.conf * 100)}% conf
                            </span>
                          </div>

                          {isMatch && catBadge && (
                            <span
                              className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold border ${catBadge.bg} ${catBadge.border} ${catBadge.text}`}
                            >
                              {catBadge.icon}
                              {catBadge.label}
                            </span>
                          )}
                        </div>

                        {/* Plate & Multi-frame Consensus */}
                        <div className="flex items-center justify-between pt-1 border-t border-police-800/60">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-400 uppercase font-medium">Plate:</span>
                            {veh.plate_number ? (
                              <span
                                className={`px-2 py-0.5 rounded font-mono font-bold text-xs tracking-wider border ${
                                  isMatch
                                    ? "bg-red-950 border-red-600 text-red-200"
                                    : "bg-emerald-950 border-emerald-600 text-emerald-300"
                                }`}
                              >
                                {veh.plate_number}
                              </span>
                            ) : (
                              <span className="text-[10px] text-slate-500 font-mono">
                                {veh.has_plate ? "Scanning plate..." : "No reliable detection"}
                              </span>
                            )}
                          </div>

                          {veh.plate_number && veh.plate_conf > 0 && (
                            <span className="text-[10px] font-mono text-cyan-400">
                              {Math.round(veh.plate_conf * 100)}% OCR
                            </span>
                          )}
                        </div>

                        {/* Snapshot thumbnail if available */}
                        {veh.snapshot_url && (
                          <div className="pt-1 flex items-center gap-2">
                            <img
                              src={getEvidenceUrl(veh.snapshot_url)}
                              alt={veh.track_id}
                              className="w-16 h-10 object-cover rounded border border-police-700"
                              onError={(e) => {
                                ;(e.target as HTMLElement).style.display = "none"
                              }}
                            />
                            {veh.plate_snapshot_url && (
                              <img
                                src={getEvidenceUrl(veh.plate_snapshot_url)}
                                alt="Plate crop"
                                className="h-6 w-20 object-contain bg-black rounded border border-police-700"
                                onError={(e) => {
                                  ;(e.target as HTMLElement).style.display = "none"
                                }}
                              />
                            )}
                            <div className="text-[10px] text-slate-400 font-mono">
                              Consensus: {veh.frames_seen} frames
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })
              )}
            </div>
          )}

          {/* ── ALERTS TAB CONTENT ── */}
          {activeTab === "alerts" && (
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {alerts.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
                  <ShieldCheck className="w-8 h-8 mb-2 opacity-40 text-emerald-400" />
                  <p className="text-xs">No watchlist alerts triggered</p>
                  <p className="text-[10px] mt-1 text-slate-600">
                    Vehicles matching the police watchlist will trigger dynamic alerts here
                  </p>
                </div>
              ) : (
                alerts.map((alert) => {
                  const catBadge = getCategoryBadge(alert.watchlist_category)
                  const isUnack = alert.status !== "ACKNOWLEDGED"

                  return (
                    <div
                      key={alert.alert_id}
                      className={`p-3.5 rounded-xl border transition-all space-y-2.5 ${
                        isUnack
                          ? "bg-red-950/70 border-red-600 shadow-xl shadow-red-950/60"
                          : "bg-police-900/60 border-police-800 opacity-75"
                      }`}
                    >
                      {/* Top Header: Category & Priority */}
                      <div className="flex items-center justify-between">
                        <span
                          className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold border ${catBadge.bg} ${catBadge.border} ${catBadge.text}`}
                        >
                          {catBadge.icon}
                          {alert.alert_type}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-[9px] font-bold font-mono ${
                            isUnack ? "bg-red-600 text-white animate-pulse" : "bg-slate-700 text-slate-300"
                          }`}
                        >
                          {alert.status}
                        </span>
                      </div>

                      {/* Plate & Model */}
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-lg font-black text-white font-mono tracking-wider">
                            {alert.vehicle_number}
                          </p>
                          <p className="text-xs text-slate-300 font-medium mt-0.5">
                            {alert.vehicle_model}
                          </p>
                        </div>
                        <div className="text-right">
                          <span className="text-[10px] text-slate-400 block font-mono">
                            Time: {alert.detection_time}
                          </span>
                          <span className="text-[10px] font-mono text-cyan-400 block">
                            ANPR: {alert.anpr_confidence}%
                          </span>
                        </div>
                      </div>

                      {/* Camera & Confidences */}
                      <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400 font-mono bg-black/40 p-2 rounded border border-police-800/80">
                        <div>
                          <span className="text-slate-500 block">Source Camera:</span>
                          <span className="text-white truncate block">{alert.camera_name}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">Veh Detection Conf:</span>
                          <span className="text-emerald-400">{alert.vehicle_detection_confidence}%</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">Priority:</span>
                          <span className="text-red-400 font-bold">{alert.alert_priority}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">Alert ID:</span>
                          <span className="text-slate-300 truncate block">{alert.alert_id}</span>
                        </div>
                      </div>

                      {/* Evidence Snapshot Preview */}
                      {alert.evidence_frame && (
                        <div className="flex items-center gap-2 pt-1">
                          <img
                            src={getEvidenceUrl(alert.evidence_frame)}
                            alt="Evidence vehicle"
                            className="w-24 h-14 object-cover rounded border border-red-800"
                            onError={(e) => {
                              ;(e.target as HTMLElement).style.display = "none"
                            }}
                          />
                          {alert.plate_frame && (
                            <img
                              src={getEvidenceUrl(alert.plate_frame)}
                              alt="Evidence plate"
                              className="h-8 w-24 object-contain bg-black rounded border border-red-800"
                              onError={(e) => {
                                ;(e.target as HTMLElement).style.display = "none"
                              }}
                            />
                          )}
                          <div className="text-[10px] text-slate-400 flex-1">
                            {alert.case_number && <span className="block font-mono">Case: {alert.case_number}</span>}
                            <span className="text-[9px] text-slate-500">Evidence captured to disk</span>
                          </div>
                        </div>
                      )}

                      {/* Description */}
                      {alert.description && (
                        <p className="text-[11px] text-red-200/90 leading-relaxed bg-red-950/80 p-2 rounded border border-red-900/60">
                          {alert.description}
                        </p>
                      )}

                      {/* Acknowledge Button */}
                      {isUnack ? (
                        <button
                          onClick={() => handleAcknowledgeAlert(alert.alert_id)}
                          className="w-full py-1.5 rounded-lg bg-red-700 hover:bg-red-600 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                        >
                          <Check className="w-3.5 h-3.5" />
                          Acknowledge &amp; Dispatch Alert
                        </button>
                      ) : (
                        <div className="text-[10px] text-slate-400 font-mono text-center pt-1">
                          ✓ Acknowledged by {alert.acknowledged_by || "Operator"} at {alert.acknowledged_at}
                        </div>
                      )}
                    </div>
                  )
                })
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Watchlist Management Modal ──────────────────────────────────── */}
      {showWatchlistModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-police-950 border border-police-700 rounded-xl max-w-2xl w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-police-800 pb-3">
              <div className="flex items-center gap-2.5">
                <ShieldCheck className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-bold text-white">Surveillance Watchlist Configuration</h3>
              </div>
              <button
                onClick={() => setShowWatchlistModal(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Add Target Form */}
            <div className="bg-police-900/80 p-3.5 rounded-lg border border-police-800 space-y-3">
              <span className="text-xs font-bold text-white uppercase tracking-wider block">
                Add Vehicle to Active Watchlist
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                <input
                  type="text"
                  placeholder="Plate (e.g. GJ11AA1234)"
                  value={newPlateInput}
                  onChange={(e) => setNewPlateInput(e.target.value.toUpperCase())}
                  className="bg-police-950 border border-police-700 rounded px-2.5 py-1.5 text-xs text-white uppercase font-mono outline-none focus:border-cyan-500"
                />
                <input
                  type="text"
                  placeholder="Model (e.g. Hyundai Creta)"
                  value={newModelInput}
                  onChange={(e) => setNewModelInput(e.target.value)}
                  className="bg-police-950 border border-police-700 rounded px-2.5 py-1.5 text-xs text-white outline-none focus:border-cyan-500"
                />
                <select
                  value={newCatInput}
                  onChange={(e) => setNewCatInput(e.target.value)}
                  className="bg-police-950 border border-police-700 rounded px-2.5 py-1.5 text-xs text-white outline-none focus:border-cyan-500"
                >
                  <option value="STOLEN_VEHICLE">Stolen Vehicle</option>
                  <option value="WANTED_PERSON">Wanted Person</option>
                  <option value="SUSPECT_VEHICLE">Suspect Vehicle</option>
                  <option value="BLACKLISTED_VEHICLE">Blacklisted</option>
                  <option value="MISSING_PERSON_VEHICLE">Missing Person</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Case / FIR Number (e.g. FIR-GJ-2026-8831)"
                  value={newCaseInput}
                  onChange={(e) => setNewCaseInput(e.target.value)}
                  className="flex-1 bg-police-950 border border-police-700 rounded px-2.5 py-1.5 text-xs text-white font-mono outline-none focus:border-cyan-500"
                />
                <button
                  onClick={handleAddWatchlist}
                  disabled={!newPlateInput.trim()}
                  className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold rounded flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add Target
                </button>
              </div>
            </div>

            {/* Current Watchlist Entries Table */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">
                Active Watchlist Targets ({watchlistEntries.length})
              </span>
              <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1">
                {watchlistEntries.map((w) => {
                  const badge = getCategoryBadge(w.category)
                  return (
                    <div
                      key={w.registration_number}
                      className="flex items-center justify-between p-2.5 bg-police-900/60 border border-police-800 rounded-lg text-xs"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono font-black text-white text-sm tracking-wider">
                          {w.registration_number}
                        </span>
                        <span className="text-slate-300 font-medium">{w.model || "Unknown Model"}</span>
                        <span
                          className={`px-1.5 py-0.2 rounded text-[9px] font-bold border ${badge.bg} ${badge.border} ${badge.text}`}
                        >
                          {badge.label}
                        </span>
                        <span className="text-[10px] text-red-400 font-bold">{w.priority}</span>
                      </div>
                      <button
                        onClick={() => handleDeleteWatchlist(w.registration_number)}
                        className="text-slate-500 hover:text-red-400 p-1 cursor-pointer transition-colors"
                        title="Delete target"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-police-800">
              <button
                onClick={() => setShowWatchlistModal(false)}
                className="px-4 py-1.5 bg-police-800 hover:bg-police-700 text-white text-xs font-semibold rounded cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
