/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: "#fff7f0",
        pastel: {
          pink: "#ffdce5",
          rose: "#ffb3c6",
          blush: "#ffcee6"
        },
        cocoa: "#6b4f4f",
        mint: "#e8fff5"
      },
      boxShadow: {
        soft: "0 8px 30px rgba(0,0,0,0.06)",
      },
      borderRadius: {
        'xl2': '1.25rem'
      },
      fontFamily: {
        display: ['Poppins', 'system-ui', 'sans-serif'],
        body: ['Quicksand', 'system-ui', 'sans-serif']
      }
    },
  },
  plugins: [],
}