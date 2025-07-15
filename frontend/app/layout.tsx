import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { Toaster } from 'react-hot-toast';

import './globals.css';
import { QueryProvider } from '@/components/providers/query-provider';
import { AuthProvider } from '@/components/providers/auth-provider';
import { WalletProvider } from '@/contexts/WalletContext';
import { ThemeProvider } from '@/components/providers/theme-provider';
import { Navigation } from '@/components/layout/navigation';
import { Footer } from '@/components/layout/footer';

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: {
    default: 'StellarVault - AI-Powered RWA Tokenization',
    template: '%s | StellarVault'
  },
  description: 'AI-powered platform for tokenizing Real-World Assets on Stellar blockchain. Unlock the $16T illiquid asset market through fractional ownership and intelligent valuation.',
  keywords: [
    'stellar',
    'blockchain', 
    'tokenization',
    'real-world assets',
    'RWA',
    'AI',
    'defi',
    'investment',
    'fractional ownership',
    'soroban'
  ],
  authors: [
    {
      name: 'Kamal Singh',
      url: 'https://linkedin.com/in/kamal-singh7'
    }
  ],
  creator: 'StellarVault Team',
  publisher: 'StellarVault',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://stellarvault.com'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: '/',
    title: 'StellarVault - AI-Powered RWA Tokenization',
    description: 'AI-powered platform for tokenizing Real-World Assets on Stellar blockchain',
    siteName: 'StellarVault',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'StellarVault Platform'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'StellarVault - AI-Powered RWA Tokenization',
    description: 'AI-powered platform for tokenizing Real-World Assets on Stellar blockchain',
    creator: '@kamalbuilds',
    images: ['/og-image.png']
  },
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
  manifest: '/manifest.json',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION,
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <QueryProvider>
            <AuthProvider>
              <WalletProvider>
                <div className="relative flex min-h-screen flex-col">
                  <Navigation />
                  <main className="flex-1">
                    {children}
                  </main>
                  <Footer />
                </div>
                <Toaster
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    style: {
                      background: 'hsl(var(--card))',
                      color: 'hsl(var(--card-foreground))',
                      border: '1px solid hsl(var(--border))',
                    },
                    success: {
                      iconTheme: {
                        primary: 'hsl(var(--success-500))',
                        secondary: 'white',
                      },
                    },
                    error: {
                      iconTheme: {
                        primary: 'hsl(var(--error-500))',
                        secondary: 'white',
                      },
                    },
                  }}
                />
              </WalletProvider>
            </AuthProvider>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
} 