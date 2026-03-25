'use client';

import UploadPaper from '@/components/UploadPaper';
import {
    ArrowRight,
    BarChart3,
    Cpu,
    FlaskConical,
    GitBranch,
    Zap,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

const features = [
  {
    icon: FlaskConical,
    title: 'Paper Analysis',
    description:
      'Multi-agent AI system extracts models, datasets, training procedures, and evaluation metrics from research papers.',
  },
  {
    icon: Cpu,
    title: 'Code Generation',
    description:
      'Produces complete, runnable PyTorch projects with model definitions, data loaders, training loops, and configs.',
  },
  {
    icon: GitBranch,
    title: 'Architecture Diagrams',
    description:
      'Auto-generates Mermaid diagrams for model architecture, training pipeline, and data flow visualization.',
  },
  {
    icon: BarChart3,
    title: 'Validation & Scoring',
    description:
      'Validates generated code for correctness, completeness, and reproducibility with detailed quality scores.',
  },
];

export default function HomePage() {
  const router = useRouter();
  const [isUploading, setIsUploading] = useState(false);

  const handleUploadComplete = useCallback(
    (projectId: string) => {
      router.push(`/dashboard/${projectId}`);
    },
    [router]
  );

  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <div className="flex-1 flex flex-col items-center justify-center px-8 py-16">
        <div className="max-w-3xl w-full text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-sand-100 border border-sand-300 text-sand-700 text-sm mb-6">
            <Zap className="w-3.5 h-3.5" />
            <span>AI-Powered Research Paper Analysis</span>
          </div>
          <h1 className="text-5xl font-bold mb-4 text-sand-950 leading-tight">
            Transform Papers into
            <br />
            <span className="text-sand-700">Production Code</span>
          </h1>
          <p className="text-lg text-sand-500 max-w-xl mx-auto">
            Upload an ML research paper and watch ReagentAI's multi-agent
            pipeline generate complete, validated, production-ready code.
          </p>
        </div>

        {/* Upload Zone */}
        <div className="w-full max-w-2xl mb-16">
          <UploadPaper
            onUploadComplete={handleUploadComplete}
            isUploading={isUploading}
            setIsUploading={setIsUploading}
          />
        </div>

        {/* Features Grid */}
        <div className="w-full max-w-4xl">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="group p-5 rounded-xl bg-white border border-sand-200 hover:border-sand-300 hover:shadow-md transition-all duration-300"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-sand-100 border border-sand-200 flex items-center justify-center flex-shrink-0 group-hover:bg-sand-200 transition-colors">
                      <Icon className="w-5 h-5 text-sand-700" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-sand-900 mb-1">
                        {feature.title}
                      </h3>
                      <p className="text-sm text-sand-500 leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Quick Start Link */}
        <div className="mt-12">
          <button
            onClick={() => router.push('/dashboard/demo')}
            className="inline-flex items-center gap-2 text-sm text-sand-400 hover:text-sand-700 transition-colors"
          >
            <span>Or explore with a demo project</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
