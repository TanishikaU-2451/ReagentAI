import type { Metadata } from 'next';
import './globals.css';
import Link from 'next/link';
import {
  Upload,
  LayoutDashboard,
  MessageSquare,
  FlaskConical,
  Github,
} from 'lucide-react';

export const metadata: Metadata = {
  title: 'ReagentAI - ML Paper to Production Code',
  description:
    'Transform machine learning research papers into production-ready, reproducible code with AI-powered agents.',
};

const navItems = [
  { href: '/', label: 'Upload', icon: Upload },
  { href: '/dashboard/latest', label: 'Dashboard', icon: LayoutDashboard },
  { href: '#chat', label: 'Chat', icon: MessageSquare },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-white min-h-screen flex">
        {/* Sidebar Navigation */}
        <aside className="fixed left-0 top-0 h-full w-64 bg-gray-900 border-r border-gray-800 flex flex-col z-50">
          {/* Logo */}
          <div className="p-6 border-b border-gray-800">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:shadow-indigo-500/40 transition-shadow">
                <FlaskConical className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                  ReagentAI
                </h1>
                <p className="text-[10px] text-gray-500 tracking-wider uppercase">
                  Paper to Production
                </p>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="flex-1 p-4 space-y-1">
            <p className="text-[11px] text-gray-500 uppercase tracking-wider font-semibold px-3 mb-3">
              Navigation
            </p>
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800/60 transition-all duration-200 group"
                >
                  <Icon className="w-4.5 h-4.5 group-hover:text-indigo-400 transition-colors" />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-gray-800">
            <div className="flex items-center gap-2 px-3 py-2 text-gray-500 text-xs">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>System Online</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 text-gray-500 text-xs">
              <Github className="w-3.5 h-3.5" />
              <span>ReagentAI v0.13.0</span>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 ml-64 min-h-screen">{children}</main>
      </body>
    </html>
  );
}
