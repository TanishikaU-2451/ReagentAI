'use client';

import { api } from '@/lib/api';
import { CheckCircle2, Download, Loader2 } from 'lucide-react';
import { useCallback, useState } from 'react';

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
          ? 'bg-emerald-50 text-emerald-700 border border-emerald-300'
          : downloading
          ? 'bg-sand-100 text-sand-400 cursor-wait border border-sand-200'
          : 'bg-sand-100 text-sand-700 hover:bg-sand-200 hover:text-sand-900 border border-sand-200'
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
        <span className="text-[10px] text-sand-400 ml-1">({fileSize})</span>
      )}
    </button>
  );
}
