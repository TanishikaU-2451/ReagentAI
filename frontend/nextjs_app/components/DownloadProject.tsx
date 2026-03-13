'use client';

import { useState, useCallback } from 'react';
import { Download, Loader2, CheckCircle2, Package } from 'lucide-react';
import { api } from '@/lib/api';

interface DownloadProjectProps {
  projectId: string;
}

export default function DownloadProject({ projectId }: DownloadProjectProps) {
  const [downloading, setDownloading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [fileSize, setFileSize] = useState<string | null>(null);

  const handleDownload = useCallback(async () => {
    if (downloading) return;

    setDownloading(true);
    setDownloaded(false);

    try {
      const blob = await api.downloadProject(projectId);
      const size = blob.size;
      setFileSize(formatSize(size));

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `reagentai-project-${projectId}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 3000);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setDownloading(false);
    }
  }, [projectId, downloading]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <button
      onClick={handleDownload}
      disabled={downloading}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
        downloaded
          ? 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/20'
          : downloading
          ? 'bg-gray-800 text-gray-400 cursor-wait'
          : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
      }`}
    >
      {downloading ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : downloaded ? (
        <CheckCircle2 className="w-4 h-4" />
      ) : (
        <Download className="w-4 h-4" />
      )}
      <span>
        {downloading
          ? 'Downloading...'
          : downloaded
          ? 'Downloaded!'
          : 'Download'}
      </span>
      {fileSize && !downloading && (
        <span className="text-[10px] text-gray-500 ml-1">({fileSize})</span>
      )}
    </button>
  );
}
