/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Environment variables
  env: {
    NEXT_PUBLIC_APP_NAME: 'StellarVault',
    NEXT_PUBLIC_APP_VERSION: '1.0.0',
  },

  // Image optimization
  images: {
    domains: [
      'localhost',
      'stellarvault.com',
      'assets.stellarvault.com',
      'cdn.stellarvault.com',
      // Add other image domains as needed
    ],
    formats: ['image/webp', 'image/avif'],
  },

  // Experimental features
  experimental: {
    // Enable Server Components
    serverComponentsExternalPackages: ['@stellar/stellar-sdk'],
  },

  // Webpack configuration
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Handle node modules that need polyfills
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      net: false,
      tls: false,
    };

    // Add source maps in development
    if (dev) {
      config.devtool = 'cheap-module-source-map';
    }

    return config;
  },

  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ];
  },

  // Redirects
  async redirects() {
    return [
      {
        source: '/login',
        destination: '/auth/login',
        permanent: true,
      },
      {
        source: '/register',
        destination: '/auth/register',
        permanent: true,
      },
    ];
  },

  // API rewrites for development
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/api/ai/:path*',
        destination: 'http://localhost:8001/api/v1/:path*',
      },
    ];
  },

  // Build optimization
  compiler: {
    // Remove console.log in production
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Output configuration
  output: 'standalone',

  // TypeScript configuration
  typescript: {
    // Allow production builds to successfully complete even if there are type errors
    ignoreBuildErrors: true,
  },

  // ESLint configuration
  eslint: {
    // Allow production builds to successfully complete even if there are lint errors
    ignoreDuringBuilds: true,
  },

  // Compression
  compress: true,

  // Power-ups for development
  poweredByHeader: false,

  // Trailing slash
  trailingSlash: false,

  // Base path (for subdirectory deployments)
  // basePath: '/stellarvault',

  // Asset prefix (for CDN)
  // assetPrefix: 'https://cdn.stellarvault.com',
};

// Bundle analyzer
if (process.env.ANALYZE === 'true') {
  const withBundleAnalyzer = require('@next/bundle-analyzer')({
    enabled: true,
  });
  module.exports = withBundleAnalyzer(nextConfig);
} else {
  module.exports = nextConfig;
} 