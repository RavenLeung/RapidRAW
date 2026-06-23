import { useState, useEffect, useCallback } from 'react';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow';

interface ReferenceImageData {
  dataUrl: string;
}

export default function ReferenceViewer() {
  const [imageData, setImageData] = useState<string | null>(null);
  const [photoLabel, setPhotoLabel] = useState<string>('');

  const closeWindow = useCallback(() => {
    const win = getCurrentWebviewWindow();
    win.close().catch(() => window.close());
  }, []);

  useEffect(() => {
    const unlistenUpdate = listen<ReferenceImageData>('reference:update-image', (event) => {
      setImageData(event.payload.dataUrl);
      setPhotoLabel(event.payload.label || '');
    });

    const unlistenClose = listen('reference:close', () => {
      closeWindow();
    });

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeWindow();
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      unlistenUpdate.then((fn) => fn());
      unlistenClose.then((fn) => fn());
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [closeWindow]);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      {imageData ? (
        <img
          src={imageData}
          alt={photoLabel || 'Reference view'}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            userSelect: 'none',
            WebkitUserSelect: 'none',
          }}
          draggable={false}
        />
      ) : (
        <span
          style={{
            color: '#555',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            fontSize: 18,
          }}
        >
          Waiting for image...
        </span>
      )}

      {/* Hint overlay */}
      <span
        style={{
          position: 'fixed',
          bottom: 24,
          left: '50%',
          transform: 'translateX(-50%)',
          color: 'rgba(255,255,255,0.25)',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          fontSize: 13,
          pointerEvents: 'none',
          userSelect: 'none',
        }}
      >
        Press Esc to close
      </span>
    </div>
  );
}
