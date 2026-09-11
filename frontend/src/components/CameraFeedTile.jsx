import { useEffect, useRef } from 'react';
import { feedSnapshotUrl } from '../api/endpoints';

/** Snapshot polling avoids browser 6-connection limit (MJPEG breaks with 12+ cameras). */
export default function CameraFeedTile({ cameraId, index = 0, alt = '' }) {
  const imgRef = useRef(null);
  const blobUrlRef = useRef(null);
  const busyRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let intervalId = null;
    const staggerMs = (index % 12) * 80;
    const periodMs = 100;

    const poll = async () => {
      if (cancelled || busyRef.current) return;
      busyRef.current = true;
      try {
        const res = await fetch(`${feedSnapshotUrl(cameraId)}?_=${Date.now()}`, { credentials: 'include' });
        if (cancelled || !res.ok || res.status === 204) return;
        const blob = await res.blob();
        if (cancelled || !blob.size) return;
        const objUrl = URL.createObjectURL(blob);
        const img = imgRef.current;
        if (img && !cancelled) {
          const old = blobUrlRef.current;
          img.src = objUrl;
          blobUrlRef.current = objUrl;
          if (old) URL.revokeObjectURL(old);
        } else {
          URL.revokeObjectURL(objUrl);
        }
      } catch {
        /* transient */
      } finally {
        busyRef.current = false;
      }
    };

    const startId = setTimeout(() => {
      poll();
      intervalId = setInterval(poll, periodMs);
    }, staggerMs);

    return () => {
      cancelled = true;
      clearTimeout(startId);
      if (intervalId) clearInterval(intervalId);
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, [cameraId, index]);

  return <img ref={imgRef} alt={alt} className="live-feed" draggable={false} />;
}
