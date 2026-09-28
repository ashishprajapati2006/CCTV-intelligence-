import {
  Camera,
  CameraHealth,
  StreamDescriptor,
  ObservedVehicle,
  VehicleObservation,
  JourneyDossier,
  WatchlistEntry,
  Alert,
  DashboardStats,
  AnalyticsData
} from "../types"

export const API_BASE = (
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== "undefined" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1"
    ? "https://cctv-intelligence-platform-backend.onrender.com"
    : "")
).replace(/\/$/, "")


export function getEvidenceUrl(url?: string): string | undefined {
  if (!url) return undefined
  if (url.startsWith("http://") || url.startsWith("https://")) return url
  return `${API_BASE}${url}`
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const isCrossOrigin = Boolean(API_BASE && typeof window !== "undefined" && !API_BASE.startsWith(window.location.origin))
  const res = await fetch(`${API_BASE}${endpoint}`, {
    credentials: isCrossOrigin ? "include" : "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  })

  if (!res.ok) {
    const errorText = await res.text().catch(() => res.statusText)
    throw new Error(`API error ${res.status}: ${errorText}`)
  }

  const contentType = res.headers.get("content-type") || ""
  if (contentType.includes("text/html")) {
    throw new Error(`API returned HTML document instead of JSON for ${endpoint}`)
  }

  return res.json()
}

export const api = {
  // Cameras
  getCameras: (search?: string, department?: string) => {
    const params = new URLSearchParams()
    if (search) params.append("search", search)
    if (department) params.append("department", department)
    const qs = params.toString()
    return request<Camera[]>(`/api/cameras${qs ? `?${qs}` : ""}`)
  },
  getCamera: (id: string) => request<Camera>(`/api/cameras/${id}`),
  getCameraHealth: (id: string) => request<CameraHealth>(`/api/cameras/${id}/health`),
  getCameraStreams: (id: string) => request<StreamDescriptor[]>(`/api/cameras/${id}/streams`),
  getCameraShortestPath: (fromCamera: string, toCamera: string) =>
    request<any>(
      `/api/cameras/shortest-path?from_camera=${encodeURIComponent(fromCamera)}&to_camera=${encodeURIComponent(toCamera)}`
    ),

  // Vehicles
  getVehicles: (params?: { search?: string; cameraId?: string; minConsensus?: number; status?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams()
    if (params?.search) qs.append("search", params.search)
    if (params?.cameraId) qs.append("camera_id", params.cameraId)
    if (params?.minConsensus !== undefined) qs.append("min_consensus", params.minConsensus.toString())
    if (params?.status) qs.append("status", params.status)
    if (params?.limit) qs.append("limit", params.limit.toString())
    if (params?.offset) qs.append("offset", params.offset.toString())
    const q = qs.toString()
    return request<ObservedVehicle[]>(`/api/vehicles${q ? `?${q}` : ""}`)
  },
  getVehicle: (reg: string) => request<ObservedVehicle>(`/api/vehicles/${encodeURIComponent(reg)}`),
  getVehicleObservations: (reg: string) => request<VehicleObservation[]>(`/api/vehicles/${encodeURIComponent(reg)}/observations`),
  getVehicleJourney: (reg: string) => request<JourneyDossier>(`/api/vehicles/${encodeURIComponent(reg)}/journey`),
  getVehicleGeojson: (reg: string) => request<any>(`/api/vehicles/${encodeURIComponent(reg)}/geojson`),

  // Watchlist
  getWatchlist: (params?: { search?: string; category?: string; priority?: string; status?: string }) => {
    const qs = new URLSearchParams()
    if (params?.search) qs.append("search", params.search)
    if (params?.category) qs.append("category", params.category)
    if (params?.priority) qs.append("priority", params.priority)
    if (params?.status) qs.append("status", params.status)
    const q = qs.toString()
    return request<WatchlistEntry[]>(`/api/watchlist${q ? `?${q}` : ""}`)
  },
  addWatchlistTarget: (target: { registration_number: string; category?: string; priority?: string; description?: string; notes?: string }) => {
    return request<WatchlistEntry>("/api/watchlist", {
      method: "POST",
      body: JSON.stringify(target),
    })
  },

  // Alerts
  getAlerts: (params?: { status?: string; priority?: string; cameraId?: string; search?: string }) => {
    const qs = new URLSearchParams()
    if (params?.status) qs.append("status", params.status)
    if (params?.priority) qs.append("priority", params.priority)
    if (params?.cameraId) qs.append("camera_id", params.cameraId)
    if (params?.search) qs.append("search", params.search)
    const q = qs.toString()
    return request<Alert[]>(`/api/alerts${q ? `?${q}` : ""}`)
  },
  getAlert: (id: string) => request<Alert>(`/api/alerts/${id}`),
  updateAlertAction: (id: string, action: "acknowledge" | "escalate" | "resolve" | "dismiss", operator: string, notes?: string) => {
    return request<Alert>(`/api/alerts/${id}/action`, {
      method: "POST",
      body: JSON.stringify({ action, operator, notes }),
    })
  },

  // Dashboard
  getDashboardStats: () => request<DashboardStats>("/api/dashboard/stats"),
  getDashboardAnalytics: () => request<AnalyticsData>("/api/dashboard/analytics"),
}
