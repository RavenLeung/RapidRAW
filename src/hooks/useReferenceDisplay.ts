import { useCallback, useEffect, useRef, useState } from 'react';
import { emit } from '@tauri-apps/api/event';
import { WebviewWindow } from '@tauri-apps/api/webviewWindow';
import { useShallow } from 'zustand/react/shallow';
import { useEditorStore } from '../store/useEditorStore';

const REF_WINDOW_LABEL = 'reference';

function blobToBase64(blobUrl: string): Promise<string> {
  return fetch(blobUrl)
    .then((r) => r.blob())
    .then(
      (blob) =>
        new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        }),
    );
}

function buildReferenceUrl(): string {
  const url = new URL(window.location.href);
  url.searchParams.set('view', 'reference');
  return url.href;
}

export function useReferenceDisplay() {
  const [isOpen, setIsOpen] = useState(false);
  const refWindowRef = useRef<WebviewWindow | null>(null);
  const isOpenRef = useRef(false);
  const throttleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingUpdateRef = useRef<string | null>(null);

  const finalPreviewUrl = useEditorStore((s) => s.finalPreviewUrl);
  const selectedImagePath = useEditorStore((s) => s.selectedImage?.path ?? null);

  const flushPendingUpdate = useCallback(async () => {
    if (!pendingUpdateRef.current || !isOpenRef.current) return;
    const dataUrl = pendingUpdateRef.current;
    pendingUpdateRef.current = null;
    try {
      await emit('reference:update-image', { dataUrl });
    } catch (err) {
      console.warn('Reference: failed to send update', err);
    }
  }, []);

  const pushImageUpdate = useCallback(
    async (blobUrl: string) => {
      if (!isOpenRef.current) return;
      try {
        const dataUrl = await blobToBase64(blobUrl);
        pendingUpdateRef.current = dataUrl;

        // Debounce: only send after 200ms of silence
        if (throttleTimerRef.current) {
          clearTimeout(throttleTimerRef.current);
        }
        throttleTimerRef.current = setTimeout(() => {
          flushPendingUpdate();
        }, 200);
      } catch (err) {
        console.warn('Reference: failed to convert blob', err);
      }
    },
    [flushPendingUpdate],
  );

  // Auto-update when finalPreviewUrl changes
  useEffect(() => {
    if (finalPreviewUrl && isOpenRef.current) {
      pushImageUpdate(finalPreviewUrl);
    }
  }, [finalPreviewUrl, pushImageUpdate]);

  const openReferenceWindow = useCallback(async () => {
    if (isOpenRef.current) return;

    const refUrl = buildReferenceUrl();

    const win = new WebviewWindow(REF_WINDOW_LABEL, {
      url: refUrl,
      title: 'Reference Display',
      width: 800,
      height: 600,
      minWidth: 400,
      minHeight: 300,
      decorations: true,
      focus: false,
    });

    // Wait for window to be created, then send first frame
    win.once('tauri://created', () => {
      isOpenRef.current = true;
      setIsOpen(true);
      refWindowRef.current = win;

      // Push current image if available
      const currentUrl = useEditorStore.getState().finalPreviewUrl;
      if (currentUrl) {
        pushImageUpdate(currentUrl);
      }
    });

    win.once('tauri://error', (e) => {
      console.error('Reference window failed to create:', e);
      isOpenRef.current = false;
      setIsOpen(false);
      refWindowRef.current = null;
    });

    // Listen for window close (user manually closes)
    // Tauri v2: listen on the window for close-requested
    try {
      const unlisten = await win.onCloseRequested(() => {
        isOpenRef.current = false;
        setIsOpen(false);
        refWindowRef.current = null;
        unlisten();
      });
    } catch (err) {
      console.warn('Reference: onCloseRequested not available', err);
    }
  }, [pushImageUpdate]);

  const closeReferenceWindow = useCallback(async () => {
    if (refWindowRef.current) {
      try {
        await emit('reference:close');
        await refWindowRef.current.close();
      } catch {
        // window might already be closed
      }
    }
    isOpenRef.current = false;
    setIsOpen(false);
    refWindowRef.current = null;
    if (throttleTimerRef.current) {
      clearTimeout(throttleTimerRef.current);
      throttleTimerRef.current = null;
    }
    pendingUpdateRef.current = null;
  }, []);

  const toggleReferenceWindow = useCallback(async () => {
    if (isOpenRef.current) {
      await closeReferenceWindow();
    } else {
      await openReferenceWindow();
    }
  }, [openReferenceWindow, closeReferenceWindow]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isOpenRef.current) {
        closeReferenceWindow();
      }
    };
  }, [closeReferenceWindow]);

  return {
    isReferenceOpen: isOpen,
    toggleReferenceWindow,
    openReferenceWindow,
    closeReferenceWindow,
  };
}
