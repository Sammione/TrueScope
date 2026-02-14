import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ESG Intelligence Hub | Advanced Greenwashing Detection",
  description: "Verify corporate ESG claims with state-of-the-art AI analysis and evidence-backed verification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${outfit.variable} antialiased`}>
        <div className="aurora-bg">
          <div className="aurora-blob bg-emerald-500 w-[600px] h-[600px] -top-48 -left-48" />
          <div className="aurora-blob bg-teal-500 w-[500px] h-[500px] top-1/2 -right-24" />
          <div className="aurora-blob bg-sky-500 w-[400px] h-[400px] -bottom-24 left-1/3" />
        </div>
        {children}
      </body>
    </html>
  );
}
