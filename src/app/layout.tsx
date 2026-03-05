import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Who's Your Doppelganger?",
  description:
    "Find your celebrity lookalike using facial recognition — right in your browser.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
