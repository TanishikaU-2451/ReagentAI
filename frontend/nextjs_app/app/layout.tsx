import {
    FlaskConical,
    Github,
    LayoutDashboard,
    MessageSquare,
    Upload,
} from 'lucide-react';
import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

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
    <html lang="en">
      <body className="bg-[var(--background)] text-[var(--foreground)] min-h-screen flex">
        {/* Sidebar Navigation */}
        <aside className="fixed left-0 top-0 h-full w-64 bg-[var(--sidebar)] border-r border-sand-800/30 flex flex-col z-50">
          {/* Logo */}
          <div className="p-6 border-b border-sand-800/20">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-sand-700 rounded-xl flex items-center justify-center shadow-lg group-hover:bg-sand-600 transition-colors">
                <FlaskConical className="w-5 h-5 text-sand-100" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-sand-100">
                  ReagentAI
                </h1>
                <p className="text-[10px] text-sand-500 tracking-wider uppercase">
                  Paper to Production
                </p>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="flex-1 p-4 space-y-1">
            <p className="text-[11px] text-sand-500 uppercase tracking-wider font-semibold px-3 mb-3">
              Navigation
            </p>
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sand-400 hover:text-sand-100 hover:bg-sand-800/40 transition-all duration-200 group"
                >
                  <Icon className="w-4.5 h-4.5 group-hover:text-sand-200 transition-colors" />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-sand-800/20">
            <div className="flex items-center gap-2 px-3 py-2 text-sand-500 text-xs">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>System Online</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 text-sand-500 text-xs">
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
