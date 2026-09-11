import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import CamerasPage from './pages/CamerasPage';
import AiConfigPage from './pages/AiConfigPage';
import DashboardPage from './pages/DashboardPage';
import PlantMapPage from './pages/PlantMapPage';
import AlertsPage from './pages/AlertsPage';
import AssignedEngineersPage from './pages/AssignedEngineersPage';
import SystemPage from './pages/SystemPage';
import EntryExitZonePage from './pages/EntryExitZonePage';
import FlapgateZonePage from './pages/FlapgateZonePage';
import WorkforceZonePage from './pages/WorkforceZonePage';
import IntrusionZonePage from './pages/IntrusionZonePage';

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route index element={<CamerasPage />} />
            <Route path="ai-config" element={<AiConfigPage />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="plant-map" element={<PlantMapPage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="assigned-engineers" element={<AssignedEngineersPage />} />
            <Route path="system" element={<SystemPage />} />
            <Route path="zone-config/:cameraId" element={<EntryExitZonePage />} />
            <Route path="flapgate-zone-config/:cameraId" element={<FlapgateZonePage />} />
            <Route path="workforce-zone-config/:cameraId" element={<WorkforceZonePage />} />
            <Route path="intrusion-zone-config/:cameraId" element={<IntrusionZonePage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
