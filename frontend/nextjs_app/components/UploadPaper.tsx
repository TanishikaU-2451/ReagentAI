'use client';

import { useState, useRef, useCallback, DragEvent, ChangeEvent } from 'react';
import { Upload, FileText, X, Loader2, CheckCircle2 } from 'lucide-react';
import { api } from '@/lib/api';

interface UploadPaperProps {
  onUploadComplete: (projectId: string) => void;
  isUploading: boolean;
  setIsUploading: (uploading: boolean) => void;
}

export default function UploadPaper({
  onUploadComplete,
  isUploading,
  setIsUploading,
}: UploadPaperProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported. Please select a .pdf file.');
      return false;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('File is too large. Maximum size is 100MB.');
      return false;
    }
    setError(null);
    return true;
  };

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  }, []);

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  }, []);

  const handleRemoveFile = useCallback(() => {
    setSelectedFile(null);
    setError(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Simulate progress updates while uploading
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 15;
        });
      }, 300);

      const result = await api.uploadPaper(selectedFile);

      clearInterval(progressInterval);
      setUploadProgress(100);

      // Brief delay to show completion
      setTimeout(() => {
        onUploadComplete(result.project_id);
      }, 500);
    } catch (err: any) {
      setError(err.message || 'Upload failed. Please try again.');
      setUploadProgress(0);
    } finally {
      setIsUploading(false);
    }
  }, [selectedFile, setIsUploading, onUploadComplete]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="w-full">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && fileInputRef.current?.click()}
        className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer ${
          isDragging
            ? 'border-indigo-500 bg-indigo-500/10 drop-zone-active'
            : selectedFile
            ? 'border-gray-700 bg-gray-900/50'
            : 'border-gray-700 bg-gray-900/30 hover:border-gray-600 hover:bg-gray-900/50'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="p-8 flex flex-col items-center justify-center min-h-[220px]">
          {!selectedFile ? (
            <>
              <div
                className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-all ${
                  isDragging
                    ? 'bg-indigo-500/20 scale-110'
                    : 'bg-gray-800'
                }`}
              >
                <Upload
                  className={`w-7 h-7 transition-colors ${
                    isDragging ? 'text-indigo-400' : 'text-gray-500'
                  }`}
                />
              </div>
              <p className="text-base font-medium text-gray-300 mb-1">
                {isDragging
                  ? 'Drop your paper here'
                  : 'Drag & drop your research paper'}
              </p>
              <p className="text-sm text-gray-500 mb-4">
                or click to browse files
              </p>
              <div className="flex items-center gap-4">
                <span className="text-xs text-gray-600 px-3 py-1 rounded-full bg-gray-800/50">
                  PDF only
                </span>
                <span className="text-xs text-gray-600 px-3 py-1 rounded-full bg-gray-800/50">
                  Max 100MB
                </span>
              </div>
            </>
          ) : (
            <>
              {/* Selected file display */}
              <div className="w-full max-w-md">
                <div className="flex items-center gap-4 p-4 rounded-xl bg-gray-800/50 border border-gray-700">
                  <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center flex-shrink-0">
                    <FileText className="w-6 h-6 text-indigo-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                  {!isUploading && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveFile();
                      }}
                      className="p-1.5 rounded-lg hover:bg-gray-700 text-gray-500 hover:text-white transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Progress bar */}
                {isUploading && (
                  <div className="mt-4">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs text-gray-400">
                        {uploadProgress >= 100
                          ? 'Upload complete!'
                          : 'Uploading...'}
                      </span>
                      <span className="text-xs text-gray-500">
                        {Math.round(uploadProgress)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-1.5">
                      <div
                        className="h-1.5 rounded-full bg-indigo-500 transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Upload button */}
                {!isUploading && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUpload();
                    }}
                    className="mt-4 w-full py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30"
                  >
                    <Upload className="w-4 h-4" />
                    Upload & Process Paper
                  </button>
                )}

                {isUploading && uploadProgress >= 100 && (
                  <div className="mt-3 flex items-center justify-center gap-2 text-emerald-400 text-sm">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Redirecting to dashboard...</span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="mt-3 flex items-center gap-2 text-red-400 text-sm px-1">
          <X className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
