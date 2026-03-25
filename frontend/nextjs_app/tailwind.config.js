/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        sand: {
          50: '#faf8f5',
          100: '#f5f0ea',
          200: '#ebe3d7',
          300: '#ddd1be',
          400: '#c9b89e',
          500: '#b5a080',
          600: '#9a8466',
          700: '#7d6a52',
          800: '#5e4f3d',
          900: '#3f352a',
          950: '#2a231c',
        },
        warm: {
          50: '#fdfcfb',
          100: '#f7f5f2',
          200: '#efeae4',
          300: '#e2dbd1',
          400: '#cfc4b5',
          500: '#b8a994',
          600: '#968472',
          700: '#756758',
          800: '#574c40',
          900: '#3b332b',
          950: '#251f19',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'spin-slow': 'spin 2s linear infinite',
        'pulse-subtle': 'pulse 3s ease-in-out infinite',
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        slideInRight: {
          '0%': { transform: 'translateX(100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
};
