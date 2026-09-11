import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { getPlantMap } from '../api/endpoints';

const icon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
});

export default function PlantMapPage() {
  const [data, setData] = useState({ cameras: [], plant_name: '' });

  useEffect(() => {
    getPlantMap().then(setData).catch(console.error);
  }, []);

  const center = data.cameras.length
    ? [data.cameras[0].latitude, data.cameras[0].longitude]
    : [12.9716, 77.5946];

  return (
    <>
      <div className="page-header">
        <h1 className="page-title gradient-text">Plant Map</h1>
        <p className="page-subtitle">{data.plant_name || 'Food processing facility camera locations'}</p>
      </div>
      <div className="map-wrap glass-panel">
        <MapContainer center={center} zoom={16} style={{ height: '520px', width: '100%', borderRadius: '16px' }}>
          <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {data.cameras.map((cam) => (
            <Marker key={cam.camera_id} position={[cam.latitude, cam.longitude]} icon={icon}>
              <Popup>
                <strong>{cam.name}</strong><br />
                {cam.location_label}<br />
                Status: {cam.status}
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
    </>
  );
}
