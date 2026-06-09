import type { Metadata } from "next";
import { Quicksand } from "next/font/google";
import "./globals.css";
import BottomNav from "../components/BottomNav";

const quicksand = Quicksand({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Signtara PWA",
  description: "Aplikasi Penerjemah Bahasa Isyarat BISINDO",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#F97316" />
      </head>
      <body className={`${quicksand.className} antialiased`}>
        <main className="max-w-md mx-auto min-h-screen bg-signtara-bg relative shadow-sm pb-24">
          {children}
          <BottomNav />
        </main>
      </body>
    </html>
  );
}