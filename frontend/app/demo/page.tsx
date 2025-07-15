'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Shield, TrendingUp, Wallet, Building, Palette, Gem, BarChart3, Globe, Users, Lock, Zap } from 'lucide-react';
import WalletConnect from '@/components/WalletConnect';
import { useWallet, WalletStatus } from '@/contexts/WalletContext';

// Mock data for demo
const mockAssets = [
  {
    id: 1,
    name: 'Manhattan Office Building',
    type: 'real_estate',
    value: '$2.5M',
    tokens: '250,000',
    price: '$10.00',
    description: 'Prime commercial real estate in downtown Manhattan',
    image: '🏢',
    roi: '+12.5%',
    risk: 'Low'
  },
  {
    id: 2,
    name: 'Picasso Artwork Collection',
    type: 'art',
    value: '$1.8M',
    tokens: '180,000',
    price: '$10.00',
    description: 'Authenticated Picasso paintings from private collection',
    image: '🎨',
    roi: '+8.3%',
    risk: 'Medium'
  },
  {
    id: 3,
    name: 'Gold Commodity Pool',
    type: 'commodity',
    value: '$5.2M',
    tokens: '520,000',
    price: '$10.00',
    description: 'Physical gold stored in secure vaults',
    image: '🏅',
    roi: '+6.8%',
    risk: 'Low'
  },
  {
    id: 4,
    name: 'Green Energy Bonds',
    type: 'bond',
    value: '$3.1M',
    tokens: '310,000',
    price: '$10.00',
    description: 'Sustainable energy infrastructure bonds',
    image: '⚡',
    roi: '+9.2%',
    risk: 'Medium'
  }
];

const features = [
  {
    icon: Shield,
    title: 'SEC Compliant',
    description: 'Full regulatory compliance with automated KYC/AML'
  },
  {
    icon: TrendingUp,
    title: 'AI Valuation',
    description: 'Real-time asset pricing using machine learning'
  },
  {
    icon: Globe,
    title: 'Global Access',
    description: 'Trade tokenized assets 24/7 across borders'
  },
  {
    icon: Lock,
    title: 'Secure Custody',
    description: 'Fireblocks integration for institutional security'
  },
  {
    icon: Users,
    title: 'Fractional Ownership',
    description: 'Access high-value assets with minimal investment'
  },
  {
    icon: Zap,
    title: 'Instant Settlement',
    description: 'Fast transactions on Stellar blockchain'
  }
];

export default function DemoPage() {
  const { status } = useWallet();
  const [selectedAsset, setSelectedAsset] = useState<typeof mockAssets[0] | null>(null);
  const [tokenizeAmount, setTokenizeAmount] = useState('1000');

  const getAssetIcon = (type: string) => {
    switch (type) {
      case 'real_estate': return Building;
      case 'art': return Palette;
      case 'commodity': return Gem;
      case 'bond': return BarChart3;
      default: return Building;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-stellar-600 to-vault-600 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Experience the Future of Asset Tokenization
          </h1>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            See how StellarVault transforms real-world assets into tradeable digital tokens 
            with AI-powered valuation and regulatory compliance.
          </p>
          <div className="flex items-center justify-center space-x-4">
            <div className="bg-white/20 px-4 py-2 rounded-lg">
              <span className="text-sm">$16T Market Size</span>
            </div>
            <div className="bg-white/20 px-4 py-2 rounded-lg">
              <span className="text-sm">Real-time Trading</span>
            </div>
            <div className="bg-white/20 px-4 py-2 rounded-lg">
              <span className="text-sm">Global Access</span>
            </div>
          </div>
        </div>
      </section>

      {/* Main Demo Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column - Wallet Connection */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center text-gray-900 dark:text-white">
                <Wallet className="w-5 h-5 mr-2 text-stellar-600" />
                Connect Wallet
              </h3>
              <WalletConnect />
            </div>

            {/* Features List */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Platform Features</h3>
              <div className="space-y-4">
                {features.map((feature, index) => {
                  const Icon = feature.icon;
                  return (
                    <div key={index} className="flex items-start space-x-3">
                      <Icon className="w-5 h-5 text-stellar-600 mt-0.5" />
                      <div>
                        <h4 className="font-medium text-sm text-gray-900 dark:text-white">{feature.title}</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">{feature.description}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column - Asset Marketplace */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-black">Tokenized Asset Marketplace</h3>
                <div className="text-sm text-gray-600">
                  Total Value: <span className="font-semibold text-stellar-600 text-green-600">$12.6M</span>
                </div>
              </div>

              {/* Asset Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {mockAssets.map((asset) => {
                  const Icon = getAssetIcon(asset.type);
                  return (
                    <div
                      key={asset.id}
                      onClick={() => setSelectedAsset(asset)}
                      className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                        selectedAsset?.id === asset.id 
                          ? 'border-stellar-500 bg-stellar-50' 
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className="text-2xl">{asset.image}</div>
                          <div>
                            <h4 className="font-medium text-sm">{asset.name}</h4>
                            <p className="text-xs text-gray-600 capitalize">{asset.type.replace('_', ' ')}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-semibold">{asset.value}</p>
                          <p className={`text-xs ${asset.roi.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
                            {asset.roi}
                          </p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-600 mb-3">{asset.description}</p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-green-600">Price: {asset.price}</span>
                        <span className={`px-2 py-1 rounded-full ${
                          asset.risk === 'Low' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {asset.risk} Risk
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Selected Asset Details */}
              {selectedAsset && (
                <div className="border-t pt-6">
                  <h4 className="font-semibold mb-4 text-black">Asset Details: {selectedAsset.name}</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-600">Total Value</p>
                      <p className="font-semibold text-green-600">{selectedAsset.value}</p>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-600">Total Tokens</p>
                      <p className="font-semibold text-green-600">{selectedAsset.tokens}</p>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-600">Token Price</p>
                      <p className="font-semibold text-green-600">{selectedAsset.price}</p>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs text-gray-600">Expected ROI</p>
                      <p className="font-semibold text-green-600">{selectedAsset.roi}</p>
                    </div>
                  </div>

                  {/* Investment Form */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-medium mb-3">Invest in {selectedAsset.name}</h5>
                    <div className="flex items-center space-x-4">
                      <div className="flex-1">
                        <label className="block text-xs text-gray-600 mb-1">Investment Amount ($)</label>
                        <input
                          type="number"
                          value={tokenizeAmount}
                          onChange={(e) => setTokenizeAmount(e.target.value)}
                          className="w-full px-3 py-2 border rounded-lg text-sm text-black"
                          placeholder="1000"
                        />
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-gray-600">Tokens to receive</p>
                        <p className="font-semibold text-green-600">{(parseInt(tokenizeAmount) / 10).toLocaleString()}</p>
                      </div>
                    </div>
                    <button
                      disabled={status !== WalletStatus.CONNECTED}
                      className={`w-full mt-4 py-2 px-4 rounded-lg font-medium transition-colors ${
                        status === WalletStatus.CONNECTED
                          ? 'bg-stellar-600 text-white hover:bg-stellar-700'
                          : 'bg-gray-300 text-gray-600 cursor-not-allowed'
                      }`}
                    >
                      {status === WalletStatus.CONNECTED ? 'Invest Now' : 'Connect Wallet to Invest'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-12 bg-white rounded-lg shadow-sm border p-8 text-center">
          <h3 className="text-2xl font-bold mb-4 text-black">Ready to Start Tokenizing?</h3>
          <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
            Join the future of asset ownership. Create your account and start tokenizing 
            real-world assets with AI-powered valuations and institutional-grade security.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <Link
              href="/register"
              className="inline-flex items-center px-6 py-3 bg-stellar-600 text-white font-semibold rounded-lg hover:bg-stellar-700 transition-colors"
            >
              Create Account
              <ArrowRight className="ml-2 w-4 h-4" />
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex items-center px-6 py-3 border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors"
            >
              Go to Dashboard
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 