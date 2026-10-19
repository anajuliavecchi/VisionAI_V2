
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionAI | Monitoramento inteligente",
  description:
    "Painel de monitoramento inteligente, reconhecimento facial e gestão de acessos VisionAI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
