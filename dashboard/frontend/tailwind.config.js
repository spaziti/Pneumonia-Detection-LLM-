/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#09090b", // zinc-950
        foreground: "#fafafa", // zinc-50
        card: "#18181b", // zinc-900
        medical: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6', // primary brand color
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
          950: '#042f2e',
        },
        primary: {
          DEFAULT: '#14b8a6', // medical-500
          foreground: '#000000',
        },
        secondary: {
          DEFAULT: '#27272a', // zinc-800
          foreground: '#fafafa',
        },
        muted: {
          DEFAULT: '#27272a', // zinc-800
          foreground: '#a1a1aa', // zinc-400
        },
        accent: {
          DEFAULT: '#2dd4bf', // medical-400
          foreground: '#09090b',
        },
        border: "#27272a", // zinc-800
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'glass-gradient': 'linear-gradient(to bottom right, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01))',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'fade-in-up': 'fadeInUp 0.6s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 10px rgba(20, 184, 166, 0.2), inset 0 0 10px rgba(20, 184, 166, 0.1)' },
          '100%': { boxShadow: '0 0 20px rgba(20, 184, 166, 0.6), inset 0 0 20px rgba(20, 184, 166, 0.2)' },
        }
      }
    },
  },
  plugins: [],
}
