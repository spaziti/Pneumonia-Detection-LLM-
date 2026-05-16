import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PneumoScan AI — Intelligent Pneumonia Detection",
  description:
    "Advanced AI-powered chest X-ray analysis with Grad-CAM explainability, uncertainty quantification, and LLM radiology report generation.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
