import React, { useEffect } from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { AuthProvider } from "./context/AuthContext"
import { ThemeProvider } from "./context/ThemeContext"
import { Layout } from "./components/layout/Layout"
import { api } from "./services/api"

// Pages
import { Dashboard } from "./pages/Dashboard"
import { Cameras } from "./pages/Cameras"
import { CameraDetails } from "./pages/CameraDetails"
import { CameraMap } from "./pages/CameraMap"
import { VideoWall } from "./pages/VideoWall"
import { Vehicles } from "./pages/Vehicles"
import { VehicleDetails } from "./pages/VehicleDetails"
import { JourneyView } from "./pages/JourneyView"
import { Watchlist } from "./pages/Watchlist"
import { Alerts } from "./pages/Alerts"
import { AlertDetails } from "./pages/AlertDetails"
import { HealthView } from "./pages/HealthView"
import { SyntheticStudio } from "./pages/SyntheticStudio"

export const App: React.FC = () => {
  useEffect(() => {
    api.getHealth()
      .then((data) => {
        if (data?.status === "healthy" || data?.status) {
          console.log("backend connected")
        }
      })
      .catch((err) => {
        console.warn("Backend connection check:", err?.message || err)
      })
  }, [])

  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="cameras" element={<Cameras />} />
              <Route path="cameras/:id" element={<CameraDetails />} />
              <Route path="map" element={<CameraMap />} />
              <Route path="videowall" element={<VideoWall />} />
              <Route path="synthetic" element={<SyntheticStudio />} />
              <Route path="vehicles" element={<Vehicles />} />
              <Route path="vehicles/:reg" element={<VehicleDetails />} />
              <Route path="journey/:reg" element={<JourneyView />} />
              <Route path="watchlist" element={<Watchlist />} />
              <Route path="alerts" element={<Alerts />} />
              <Route path="alerts/:id" element={<AlertDetails />} />
              <Route path="health" element={<HealthView />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  )
}

export default App
