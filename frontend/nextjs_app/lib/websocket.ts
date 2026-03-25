/**
 * WebSocket Hook for ReagentAI Pipeline
 *
 * Custom React hook that manages a WebSocket connection for real-time
 * pipeline updates. Includes auto-reconnect logic.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: string;
}

interface UseWebSocketReturn {
  messages: WebSocketMessage[];
  connectionStatus: ConnectionStatus;
  sendMessage: (data: any) => void;
  lastMessage: WebSocketMessage | null;
}

const WS_BASE_URL =
  typeof window !== 'undefined'
    ? (process.env.NEXT_PUBLIC_WS_URL ||
      (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
        .replace(/^http/, 'ws'))
    : 'ws://localhost:8000';

const MAX_RECONNECT_ATTEMPTS = 10;
const INITIAL_RECONNECT_DELAY = 1000; // 1 second
const MAX_RECONNECT_DELAY = 30000; // 30 seconds

export function useWebSocket(projectId: string): UseWebSocketReturn {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const [actualProjectId, setActualProjectId] = useState<string>('');

  // Resolve 'latest' to actual project ID
  useEffect(() => {
    if (projectId === 'latest') {
      fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/projects/latest`)
        .then(res => res.json())
        .then(data => {
          if (data.project_id) {
            setActualProjectId(data.project_id);
          }
        })
        .catch(err => {
          console.error('[WebSocket] Failed to resolve latest project ID:', err);
          setActualProjectId(''); // Fall back to empty
        });
    } else {
      setActualProjectId(projectId);
    }
  }, [projectId]);

  const connect = useCallback(() => {
    const targetProjectId = actualProjectId;
    if (!targetProjectId) return;

    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('connecting');

    const url = `${WS_BASE_URL}/ws/pipeline/${targetProjectId}`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        console.log(`[WebSocket] Connected to pipeline ${targetProjectId}`);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return;

        try {
          const parsed: WebSocketMessage = JSON.parse(event.data);

          // Ensure message has a type
          if (!parsed.type) {
            console.warn('[WebSocket] Received message without type:', parsed);
            return;
          }

          setMessages((prev) => [...prev, parsed]);
          setLastMessage(parsed);
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', event.data, err);
        }
      };

      ws.onclose = (event: CloseEvent) => {
        if (!mountedRef.current) return;
        wsRef.current = null;
        setConnectionStatus('disconnected');
        console.log(
          `[WebSocket] Disconnected (code: ${event.code}, reason: ${event.reason})`
        );

        // Auto-reconnect unless intentionally closed
        if (event.code !== 1000 && event.code !== 1001) {
          scheduleReconnect();
        }
      };

      ws.onerror = (event: Event) => {
        if (!mountedRef.current) return;
        console.error('[WebSocket] Error:', event);
        setConnectionStatus('error');
      };
    } catch (err) {
      console.error('[WebSocket] Failed to create connection:', err);
      setConnectionStatus('error');
      scheduleReconnect();
    }
  }, [actualProjectId]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectAttempts.current >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('[WebSocket] Max reconnect attempts reached');
      setConnectionStatus('error');
      return;
    }

    // Exponential backoff with jitter
    const delay = Math.min(
      INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttempts.current) +
        Math.random() * 1000,
      MAX_RECONNECT_DELAY
    );

    reconnectAttempts.current += 1;
    console.log(
      `[WebSocket] Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttempts.current}/${MAX_RECONNECT_ATTEMPTS})`
    );

    reconnectTimeout.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, delay);
  }, [connect]);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const payload = typeof data === 'string' ? data : JSON.stringify(data);
      wsRef.current.send(payload);
    } else {
      console.warn('[WebSocket] Cannot send message - not connected');
    }
  }, []);

  // Connect on mount / project change
  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      // Clear reconnect timer
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }

      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, []);

  // Connect when actualProjectId is resolved
  useEffect(() => {
    if (actualProjectId && mountedRef.current) {
      connect();
    }
  }, [actualProjectId, connect]);

  return {
    messages,
    connectionStatus,
    sendMessage,
    lastMessage,
  };
}

export default useWebSocket;
