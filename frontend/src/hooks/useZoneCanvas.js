import { useCallback, useEffect, useRef, useState } from 'react';
import {
  CANVAS_H,
  CANVAS_W,
  canvasClickPoint,
  drawPolygon,
} from '../utils/zoneCanvas';

export function useZoneCanvas(imageUrl, layers, onDraw) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [imageReady, setImageReady] = useState(false);

  useEffect(() => {
    if (!imageUrl) return;
    setImageReady(false);
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      setImageReady(true);
    };
    img.onerror = () => {
      imageRef.current = null;
      setImageReady(false);
    };
    img.src = imageUrl;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    if (img) ctx.drawImage(img, 0, 0, CANVAS_W, CANVAS_H);
    layers.forEach(({ points, stroke, fill, closed, label }) => {
      drawPolygon(ctx, points, stroke, fill, closed, label);
    });
    onDraw?.();
  }, [layers, onDraw]);

  useEffect(() => {
    redraw();
  }, [redraw, imageReady]);

  const handleClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !imageReady) return null;
    return canvasClickPoint(canvas, e);
  };

  return { canvasRef, imageReady, handleClick, redraw };
}

export { CANVAS_W, CANVAS_H };
